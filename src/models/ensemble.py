"""
Hybrid anomaly detection: Isolation Forest for unsupervised scoring +
XGBoost for supervised classification with Isolation Forest features as input.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Structured output for production inference."""

    consumer_id: str
    anomaly_score: float  # Isolation Forest score
    fraud_probability: float  # XGBoost probability
    risk_tier: str  # LOW, MEDIUM, HIGH, CRITICAL
    key_features: Dict[str, float]  # Top SHAP values
    explanation: str  # Human-readable summary


class HybridTheftDetector:
    """
    Production-grade hybrid detector combining:
    1. Isolation Forest: Unsupervised anomaly scoring (catches novel patterns)
    2. XGBoost: Supervised classification with IF score as feature

    Reference: "A Hybrid Machine Learning Framework for Electricity Fraud Detection" [^1^]
    """

    def __init__(
        self,
        if_contamination: float = 0.05,
        if_n_estimators: int = 100,
        xgb_params: Optional[Dict] = None,
        scale_features: bool = True,
    ):
        self.if_contamination = if_contamination
        self.if_n_estimators = if_n_estimators
        self.scale_features = scale_features

        # XGBoost defaults optimized for imbalanced fraud detection
        self.xgb_params = xgb_params or {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 10,  # Handle class imbalance (fraud is rare)
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        self.isolation_forest: Optional[IsolationForest] = None
        self.xgboost_model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, float]:
        self.feature_names = list(X.columns)

        # 1. Handle Class Imbalance (Requires imbalanced-learn in pyproject.toml)
        try:
            from imblearn.combine import SMOTETomek

            smt = SMOTETomek(random_state=42)
            X_res, y_res = smt.fit_resample(X, y)
            X, y = X_res, y_res
        except ImportError:
            logger.warning("imbalanced-learn not installed, skipping resampling")

        # 2. Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )

        # 3. Scaling
        if self.scale_features:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled, X_val_scaled = X_train.values, X_val.values

        # 4. Isolation Forest Feature Engineering
        self.isolation_forest = IsolationForest(
            n_estimators=self.if_n_estimators, contamination=self.if_contamination, random_state=42
        )
        self.isolation_forest.fit(X_train_scaled)

        # Generate scores for BOTH sets to maintain feature alignment
        if_train_scores = -self.isolation_forest.decision_function(X_train_scaled)
        if_val_scores = -self.isolation_forest.decision_function(X_val_scaled)

        # 5. Append Enhanced Features
        X_train_enhanced = np.column_stack([X_train_scaled, if_train_scores])
        X_val_enhanced = np.column_stack([X_val_scaled, if_val_scores])

        # 6. XGBoost Training - Pass X_val_enhanced to the eval_set!
        self.xgboost_model = xgb.XGBClassifier(**self.xgb_params, early_stopping_rounds=50)
        self.xgboost_model.fit(
            X_train_enhanced,
            y_train,
            eval_set=[(X_val_enhanced, y_val)],  # Fixed: No more 21 vs 20 error
            verbose=False,
        )

        # Initialize SHAP for explainability
        try:
            self.shap_explainer = shap.TreeExplainer(self.xgboost_model)
        except Exception as e:
            logger.error(f"SHAP Explainer initialization failed: {e}")
            self.shap_explainer = None

        # Validation metrics
        val_preds = self.xgboost_model.predict(X_val_enhanced)
        val_probs = self.xgboost_model.predict_proba(X_val_enhanced)[:, 1]

        metrics = {
            "val_auc": roc_auc_score(y_val, val_probs),
            "val_f1": f1_score(y_val, val_preds),
            "val_precision": np.mean(val_preds[y_val == 1]),  # Precision approximation
            "val_recall": np.mean(val_preds[y_val == 1] == y_val[y_val == 1]),
            "best_iteration": self.xgboost_model.best_iteration,
        }

        logger.info(f"Validation metrics: {metrics}")
        return metrics

    def predict(
        self, X: pd.DataFrame, consumer_ids: Optional[List[str]] = None
    ) -> List[DetectionResult]:
        """
        Production inference with full explainability.

        Args:
            X: Feature matrix
            consumer_ids: Optional list of IDs for result mapping

        Returns:
            List of DetectionResult objects
        """
        if self.xgboost_model is None or self.isolation_forest is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Preprocess
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        # Isolation Forest scoring
        if_scores = -self.isolation_forest.decision_function(X_scaled)
        if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)

        # Enhanced features
        X_enhanced = np.column_stack([X_scaled, if_scores])

        # XGBoost prediction
        fraud_probs = self.xgboost_model.predict_proba(X_enhanced)[:, 1]

        # SHAP values for explainability
        if self.shap_explainer is None:
            raise RuntimeError("SHAP explainer not initialized")
        shap_values = self.shap_explainer.shap_values(X_enhanced)

        results = []
        for i, prob in enumerate(fraud_probs):
            # Risk tier classification
            if prob < 0.3:
                tier = "LOW"
            elif prob < 0.6:
                tier = "MEDIUM"
            elif prob < 0.8:
                tier = "HIGH"
            else:
                tier = "CRITICAL"

            # Top contributing features
            if isinstance(shap_values, list):  # Binary classification
                instance_shap = shap_values[1][i]  # Class 1 (fraud) contributions
            else:
                instance_shap = shap_values[i]
            feature_names = self.feature_names or []
            top_indices = np.argsort(np.abs(instance_shap))[-5:][::-1]
            key_features = {
                feature_names[idx]
                if idx < len(feature_names)
                else "isolation_forest_score": float(instance_shap[idx])
                for idx in top_indices
            }

            result = DetectionResult(
                consumer_id=consumer_ids[i] if consumer_ids else f"consumer_{i}",
                anomaly_score=float(if_scores[i]),
                fraud_probability=float(prob),
                risk_tier=tier,
                key_features=key_features,
                explanation=self._generate_explanation(key_features, tier),
            )
            results.append(result)

        return results

    def _generate_explanation(self, key_features: Dict[str, float], tier: str) -> str:
        """Generate human-readable explanation from SHAP values."""
        top_feature = max(key_features.items(), key=lambda x: abs(x[1]))
        feature_name, impact = top_feature

        direction = "increases" if impact > 0 else "decreases"

        explanations = {
            "isolation_forest_score": f"Unsupervised anomaly detection "
            f"flagged unusual consumption patterns",
            "value__standard_deviation": f"Abnormal consumption variability {direction} fraud risk",
            'value__linear_trend__attr_"slope"': f"Consumption trend {direction} fraud risk",
            "value__longest_strike_below_mean": f"Extended low-usage "
            f"periods {direction} fraud risk",
            "domain__zero_consumption_ratio": f"Zero consumption frequency {direction} fraud risk",
            "domain__sudden_drop_flag": f"Sudden consumption drop detected",
        }

        base_exp = explanations.get(feature_name, f"Feature {feature_name} {direction} risk")
        return f"[{tier}] {base_exp} (SHAP: {impact:+.3f})"

    def save(self, path: str):
        """Serialize model artifacts."""
        artifacts = {
            "isolation_forest": self.isolation_forest,
            "xgboost_model": self.xgboost_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "xgb_params": self.xgb_params,
            "if_params": {
                "contamination": self.if_contamination,
                "n_estimators": self.if_n_estimators,
            },
        }
        joblib.dump(artifacts, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "HybridTheftDetector":
        """Load model artifacts."""
        artifacts = joblib.load(path)
        detector = cls(
            if_contamination=artifacts["if_params"]["contamination"],
            if_n_estimators=artifacts["if_params"]["n_estimators"],
            xgb_params=artifacts["xgb_params"],
        )
        detector.isolation_forest = artifacts["isolation_forest"]
        detector.xgboost_model = artifacts["xgboost_model"]
        detector.scaler = artifacts["scaler"]
        detector.feature_names = artifacts["feature_names"]
        detector.shap_explainer = shap.TreeExplainer(detector.xgboost_model)
        return detector

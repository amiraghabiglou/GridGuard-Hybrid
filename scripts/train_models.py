import argparse
import logging
import pandas as pd
from pathlib import Path
from src.models.ensemble import HybridTheftDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_pipeline(feature_path: str, model_output: str, metrics_output: str):
    """
    Orchestrates the training of the Hybrid Theft Detector.
    """
    logger.info(f"Loading features from {feature_path}...")
    df = pd.read_parquet(feature_path)

    # Separate features and labels
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column for supervised training.")

    X = df.drop(columns=['label', 'consumer_id'], errors='ignore')
    y = df['label']

    logger.info("Initializing Hybrid Detector...")
    detector = HybridTheftDetector(
        if_contamination=0.05,
        xgb_params={'n_estimators': 100, 'max_depth': 4}
    )

    logger.info("Starting training (Isolation Forest + XGBoost)...")
    metrics = detector.fit(X, y)

    # Save artifacts
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    detector.save(model_output)

    # Save metrics for CI/CD reporting
    import json
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Training complete. Model saved to {model_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-output", default="models/hybrid_detector.joblib")
    parser.add_argument("--metrics-output", default="models/metrics.json")

    args = parser.parse_args()
    train_pipeline(args.features, args.model_output, args.metrics_output)
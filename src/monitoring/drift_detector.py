"""
Data drift detection for electricity theft detection pipeline.
Monitors for seasonal changes, consumption pattern shifts, and model degradation.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Structured drift detection report."""

    feature_name: str
    drift_detected: bool
    test_type: str  # 'ks', 'psi', 'wasserstein'
    statistic: float
    p_value: Optional[float]
    threshold: float
    reference_mean: float
    current_mean: float
    percent_change: float


class ElectricityDriftMonitor:
    """
    Monitors data drift in electricity consumption patterns.
    Critical for detecting:
    1. Seasonal shifts (heatwaves, holidays)
    2. Infrastructure changes (smart meter rollout)
    3. Economic impacts (recession, industrial closures)
    4. Adversarial adaptation (fraudsters changing tactics)
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        features_to_monitor: Optional[List[str]] = None,
    ):
        """
        Initialize drift monitor with reference distribution.

        Args:
            reference_data: Baseline feature distribution (training data)
            psi_threshold: Population Stability Index threshold (0.1=stable, 0.2=moderate, >0.25=significant)
            ks_threshold: Kolmogorov-Smirnov test p-value threshold
        """
        self.reference = reference_data
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.features = features_to_monitor or reference_data.columns.tolist()

        # Calculate reference statistics
        self.reference_stats = {}
        for col in self.features:
            if col in reference_data.columns:
                self.reference_stats[col] = {
                    "mean": reference_data[col].mean(),
                    "std": reference_data[col].std(),
                    "quantiles": reference_data[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values,
                    "bins": self._calculate_bins(reference_data[col]),
                }

    def _calculate_bins(self, series: pd.Series, n_bins: int = 10) -> np.ndarray:
        """Calculate bins for PSI calculation."""
        # Use deciles for stable binning
        return np.percentile(series.dropna(), np.linspace(0, 100, n_bins + 1))

    def calculate_psi(
        self, reference: pd.Series, current: pd.Series, bins: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Population Stability Index.
        PSI < 0.1: No change
        PSI 0.1-0.2: Moderate change
        PSI > 0.25: Significant change (retrain recommended)
        """
        if bins is None:
            bins = self._calculate_bins(reference)

        # Calculate proportions in each bin
        ref_counts, _ = np.histogram(reference.dropna(), bins=bins)
        curr_counts, _ = np.histogram(current.dropna(), bins=bins)

        # Add small constant to avoid division by zero
        ref_percents = (ref_counts / len(reference)) + 1e-10
        curr_percents = (curr_counts / len(current)) + 1e-10

        # Calculate PSI
        psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
        return psi

    def detect_drift(self, current_data: pd.DataFrame) -> List[DriftReport]:
        """
        Run full drift detection suite on current data.

        Returns:
            List of drift reports for each monitored feature
        """
        reports = []

        for feature in self.features:
            if feature not in current_data.columns or feature not in self.reference.columns:
                continue

            ref_series = self.reference[feature].dropna()
            curr_series = current_data[feature].dropna()

            if len(curr_series) == 0:
                continue

            # Kolmogorov-Smirnov test (distribution shift)
            ks_stat, ks_p = stats.ks_2samp(ref_series, curr_series)

            # PSI (population stability)
            psi = self.calculate_psi(ref_series, curr_series, self.reference_stats[feature]["bins"])

            # Wasserstein distance (mean shift magnitude)
            # wasserstein = stats.wasserstein_distance(ref_series, curr_series)

            # Determine if drift detected
            ks_drift = ks_p < self.ks_threshold
            psi_drift = psi > self.psi_threshold

            # Calculate percent change in mean
            ref_mean = self.reference_stats[feature]["mean"]
            curr_mean = curr_series.mean()
            pct_change = ((curr_mean - ref_mean) / abs(ref_mean) * 100) if ref_mean != 0 else 0

            # Use most sensitive test for final decision
            drift_detected = ks_drift or psi_drift

            reports.append(
                DriftReport(
                    feature_name=feature,
                    drift_detected=drift_detected,
                    test_type="combined",
                    statistic=max(ks_stat, psi),
                    p_value=ks_p if ks_drift else None,
                    threshold=self.psi_threshold,
                    reference_mean=ref_mean,
                    current_mean=curr_mean,
                    percent_change=pct_change,
                )
            )

        return reports

    def generate_alert(self, reports: List[DriftReport]) -> Optional[Dict]:
        """Generate alert if significant drift detected."""
        significant_drifts = [r for r in reports if r.drift_detected]

        if not significant_drifts:
            return None

        # Categorize drift types
        consumption_features = [
            r
            for r in significant_drifts
            if any(x in r.feature_name for x in ["mean", "consumption", "value__"])
        ]
        variance_features = [
            r
            for r in significant_drifts
            if any(x in r.feature_name for x in ["std", "variance", "entropy"])
        ]

        alert = {
            "timestamp": datetime.now().isoformat(),
            "alert_level": "CRITICAL" if len(significant_drifts) > 5 else "WARNING",
            "total_drifted_features": len(significant_drifts),
            "drifted_features": [r.feature_name for r in significant_drifts],
            "summary": {
                "consumption_pattern_shift": len(consumption_features) > 0,
                "variability_change": len(variance_features) > 0,
                "max_percent_change": max([abs(r.percent_change) for r in significant_drifts]),
            },
            "recommended_action": self._recommend_action(significant_drifts),
        }

        return alert

    def _recommend_action(self, drifts: List[DriftReport]) -> str:
        """Recommend action based on drift severity."""
        n_drifts = len(drifts)
        max_change = max([abs(d.percent_change) for d in drifts])

        if n_drifts > 10 or max_change > 50:
            return "IMMEDIATE: Retrain model with new reference data. Significant distribution shift detected."
        elif n_drifts > 5 or max_change > 30:
            return "SCHEDULED: Plan model retraining within 1 week. Monitor inspection accuracy."
        else:
            return "MONITOR: Investigate specific drifted features. Adjust thresholds if seasonal."

    def update_reference(
        self, new_reference: pd.DataFrame, validation_period: Optional[pd.DataFrame] = None
    ):
        """Update reference distribution after retraining."""
        if validation_period is not None:
            # Validate new reference is stable
            reports = self.detect_drift(validation_period)
            if any(r.drift_detected for r in reports):
                logger.warning("New reference data shows internal drift. Use longer stable period.")

        self.reference = new_reference
        self.reference_stats = {}
        for col in self.features:
            if col in new_reference.columns:
                self.reference_stats[col] = {
                    "mean": new_reference[col].mean(),
                    "std": new_reference[col].std(),
                    "quantiles": new_reference[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values,
                    "bins": self._calculate_bins(new_reference[col]),
                }
        logger.info("Updated reference distribution for drift monitoring")

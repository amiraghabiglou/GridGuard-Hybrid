"""
Time-series feature extraction using TSFRESH.
Extracts 700+ statistical features optimized for electricity theft detection.
"""
import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters, ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ElectricityFeatureExtractor:
    """
    Production-grade feature extractor for SGCC dataset.
    Handles missing values, seasonality, and domain-specific patterns.
    """

    def __init__(
        self,
        default_fc_parameters: Optional[Dict] = None,
        n_jobs: int = 4,
        chunksize: Optional[int] = None,
    ):
        """
        Initialize extractor with optimized settings for electricity data.

        Args:
            default_fc_parameters: TSFRESH feature calculator settings
            n_jobs: Parallel processing workers
            chunksize: Processing batch size for memory efficiency
        """
        # Use Efficient settings for production (faster), Comprehensive for research
        self.fc_parameters = default_fc_parameters or EfficientFCParameters()
        self.n_jobs = n_jobs
        self.chunksize = chunksize or 1000

        # Domain-specific feature selection for electricity theft
        self.relevant_features = [
            # Trend features (theft often shows sudden drops)
            'value__linear_trend__attr_"slope"',
            'value__linear_trend__attr_"stderr"',
            # Variability features (theft reduces variance)
            "value__standard_deviation",
            "value__variance",
            "value__coefficient_of_variation",
            # Entropy features (theft creates artificial regularity)
            "value__sample_entropy",
            "value__fourier_entropy__bins_10",
            # Spike/level shift detection
            "value__maximum",
            "value__minimum",
            "value__mean",
            "value__median",
            "value__longest_strike_below_mean",
            "value__longest_strike_above_mean",
            # Autocorrelation (theft breaks natural patterns)
            "value__autocorrelation__lag_1",
            "value__autocorrelation__lag_7",  # Weekly seasonality
            "value__autocorrelation__lag_30",  # Monthly patterns
            # Quantile features (distribution changes)
            "value__quantile__q_0.1",
            "value__quantile__q_0.9",
            "value__ratio_beyond_r_sigma__r_2",
            # Complexity features
            "value__cid_ce__normalize_True",
            "value__friedrich_coefficients__m_3__r_30__coeff_0",
        ]

    def prepare_data(
        self,
        df: pd.DataFrame,
        consumer_id_col: str = "consumer_id",
        date_col: str = "date",
        value_col: str = "consumption",
    ) -> pd.DataFrame:
        """
        Convert wide-format SGCC data to TSFRESH long format.

        SGCC format: consumer_id | date_1 | date_2 | ... | date_1035 | label
        TSFRESH format: id | time | value
        """
        # Handle missing values - forward fill then backward fill per consumer
        df = df.copy()

        # Melt from wide to long format
        id_vars = [consumer_id_col, "label"] if "label" in df.columns else [consumer_id_col]
        value_vars = [col for col in df.columns if col not in id_vars]

        long_df = pd.melt(
            df, id_vars=id_vars, value_vars=value_vars, var_name="time", value_name="value"
        )

        # Convert time to numeric index for TSFRESH
        long_df["time"] = pd.to_datetime(long_df["time"], errors="coerce")
        long_df = long_df.dropna(subset=["time"])
        long_df["time"] = long_df.groupby(consumer_id_col).cumcount()

        # Handle missing consumption values
        long_df["value"] = long_df.groupby(consumer_id_col)["value"].transform(
            lambda x: x.fillna(method="ffill").fillna(method="bfill").fillna(0)
        )

        return long_df.rename(columns={consumer_id_col: "id"})

    def extract_features(
        self, df_long: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Extract TSFRESH features with imputation and optional selection.

        Args:
            df_long: Data in TSFRESH format (id, time, value)
            y: Optional target for feature selection

        Returns:
            DataFrame with extracted features per consumer
        """
        logger.info(f"Extracting features for {df_long['id'].nunique()} consumers...")

        # Extract features with efficient settings
        extracted = extract_features(
            df_long,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=self.fc_parameters,
            n_jobs=self.n_jobs,
            chunksize=self.chunksize,
            show_warnings=False,
            disable_progressbar=True,
        )

        # Impute missing values (TSFRESH can produce NaNs for constant series)
        extracted = impute(extracted)

        # Filter to domain-relevant features if they exist
        available_features = [f for f in self.relevant_features if f in extracted.columns]
        if available_features:
            extracted = extracted[available_features]

        logger.info(f"Extracted {len(extracted.columns)} features")

        # Feature selection if labels provided
        if y is not None:
            extracted = self.select_relevant_features(extracted, y)

        return extracted

    def select_relevant_features(
        self, X: pd.DataFrame, y: pd.Series, fdr_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Select statistically relevant features using FDR correction.
        Critical for high-dimensional time-series features.
        """
        logger.info(f"Selecting features with FDR level {fdr_level}...")
        selected = select_features(X, y, fdr_level=fdr_level)
        logger.info(f"Selected {len(selected.columns)} / {len(X.columns)} features")
        return selected

    def add_domain_features(self, df_features: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Add hand-crafted domain features for electricity theft.
        These capture specific theft patterns not caught by generic TSFRESH.
        """
        # Calculate zero-consumption streaks (common theft pattern)
        consumption_cols = [c for c in df_raw.columns if c not in ["consumer_id", "label"]]

        # Zero consumption ratio
        df_features["domain__zero_consumption_ratio"] = (
            (df_raw[consumption_cols] == 0).mean(axis=1).values
        )

        # Sudden drop detection (consumption drops >50% and stays low)
        def detect_sudden_drop(row):
            vals = row[consumption_cols].values
            if len(vals) < 30:
                return 0
            # Rolling mean comparison
            early_mean = np.mean(vals[:30])
            late_mean = np.mean(vals[-30:])
            if early_mean > 0 and late_mean / early_mean < 0.5:
                return 1
            return 0

        df_features["domain__sudden_drop_flag"] = df_raw.apply(detect_sudden_drop, axis=1)

        # Weekend vs weekday pattern disruption (business hour theft)
        # Note: Requires date parsing, simplified here
        df_features["domain__consumption_volatility"] = df_raw[consumption_cols].std(axis=1).values

        return df_features

"""
End-to-end data pipeline for Electricity Theft Detection.
Handles raw SGCC data ingestion, cleaning, and TSFRESH feature extraction.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.features.extractors import ElectricityFeatureExtractor

# Configure logging to provide visibility during CI/CD runs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_data_pipeline(input_path: str, output_path: str, extract_tsfresh: bool = True):
    """
    Executes the data processing pipeline.

    1. Loads raw CSV (SGCC format)
    2. Cleans and handles missing values
    3. (Optional) Extracts TSFRESH features
    4. Saves to high-performance Parquet format for training
    """
    logger.info(f"🚀 Starting data pipeline. Input: {input_path}")

    # 1. Load Raw Data
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)

    df_raw = pd.read_csv(input_path)
    logger.info(f"Loaded data with shape: {df_raw.shape}")

    # 2. Initialize Extractor
    # We use n_jobs=1 in CI/CD environments to avoid memory overhead
    extractor = ElectricityFeatureExtractor(n_jobs=1)

    # 3. Cleaning and Preparation
    # SGCC data often has missing values that break statistical calculations
    logger.info("Cleaning data and handling missing values...")
    df_long = extractor.prepare_data(df_raw)

    # 4. Feature Extraction
    if extract_tsfresh:
        logger.info("Extracting TSFRESH features (this may take a few minutes)...")
        # Extract the core statistical features
        features = extractor.extract_features(df_long)

        # Add the hand-crafted domain features (Zero-consumption, drops, etc.)
        logger.info("Adding domain-specific theft features...")
        final_df = extractor.add_domain_features(features, df_raw)
    else:
        logger.info("Skipping TSFRESH; providing cleaned long-format data.")
        final_df = df_long

    # 5. Preserve Labels
    # Ensure 'label' is re-attached to the feature set if it exists in raw data
    if "label" in df_raw.columns:
        labels = df_raw.set_index("consumer_id")["label"]
        final_df = final_df.join(labels, how="left")

    # 6. Save Output
    # We use Parquet for production because it preserves schema and is 10x faster than CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path)
    logger.info(f"✅ Pipeline complete. Processed data saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridGuard Data Pipeline")
    parser.add_argument("--input", default="data/raw/sgcc.csv", help="Path to raw SGCC CSV")
    parser.add_argument(
        "--output",
        default="data/processed/features.parquet",
        help="Path to save processed features",
    )
    parser.add_argument(
        "--extract-tsfresh",
        action="store_true",
        default=True,
        help="Whether to perform TSFRESH extraction",
    )

    args = parser.parse_args()

    try:
        run_data_pipeline(
            input_path=args.input, output_path=args.output, extract_tsfresh=args.extract_tsfresh
        )
    except Exception as e:
        logger.error(f"Pipeline failed with an unexpected error: {e}")
        exit(1)

import argparse
import os

import numpy as np
import pandas as pd


def generate_mock_sgcc_data(output_path: str):
    """
    Since SGCC is a private/Kaggle dataset, this script provides
    a structure-compatible mock dataset to allow CI/CD testing.
    """
    print("Generating structure-compatible mock SGCC data for CI testing...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # SGCC Structure: consumer_id | 1035 daily columns | label
    num_consumers = 100
    num_days = 1035

    dates = [f"2014-01-{i:02d}" for i in range(1, 32)]  # Simplified dates for mock
    # Fill remaining columns to match 1035 days
    col_names = [f"day_{i}" for i in range(num_days)]

    data = np.random.uniform(0, 15, size=(num_consumers, num_days))
    labels = np.random.choice([0, 1], size=num_consumers, p=[0.9, 0.1])

    df = pd.DataFrame(data, columns=col_names)
    df.insert(0, "consumer_id", [f"CONS_{i:04d}" for i in range(num_consumers)])
    df["label"] = labels

    df.to_csv(output_path, index=False)
    print(f"Mock data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/sgcc.csv")
    args = parser.parse_args()
    generate_mock_sgcc_data(args.output)

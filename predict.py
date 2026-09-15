#!/usr/bin/env python3
"""Apply a fitted toxicity model to new five-channel OCP recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.toxicity_ml import FeatureConfig, build_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict non-negative TU values.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    metadata = pd.read_csv(args.metadata)
    config = FeatureConfig(**bundle["feature_config"])
    features, audit = build_feature_table(
        metadata,
        args.raw_dir,
        time_column=bundle["time_column"],
        channel_map=bundle["channel_map"],
        config=config,
        require_tu=False,
    )
    columns = bundle["feature_names"]
    values = features.loc[:, columns].to_numpy(dtype=float)
    predictions = np.maximum(bundle["regressor"].predict(values), 0.0)
    output = pd.DataFrame(
        {"sample_id": features["sample_id"].astype(str), "predicted_tu": predictions}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    audit.to_csv(args.output.with_name(f"{args.output.stem}_sample_accounting.csv"), index=False)


if __name__ == "__main__":
    main()

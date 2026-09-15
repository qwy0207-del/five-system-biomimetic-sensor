#!/usr/bin/env python3
"""Run the manuscript's complete machine-learning analysis without plotting."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from src.toxicity_ml import (
    CHANNELS,
    FeatureConfig,
    build_feature_table,
    evaluate_workflow,
    feature_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OCP features and reproduce the locked ML workflow."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", type=Path, help="CSV sample manifest for raw OCP files")
    source.add_argument("--features", type=Path, help="Precomputed 15-feature CSV")
    parser.add_argument("--raw-dir", type=Path, help="Directory containing raw OCP CSV files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-column", default="time_s")
    parser.add_argument(
        "--channel-columns",
        nargs=5,
        metavar=("NEURAL", "METABOLIC", "CIRCULATORY", "IMMUNE", "DEVELOPMENTAL"),
        default=list(CHANNELS),
        help="Raw CSV columns in the fixed biological-system order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = FeatureConfig()
    channel_map = dict(zip(CHANNELS, args.channel_columns, strict=True))

    if args.metadata is not None:
        if args.raw_dir is None:
            raise SystemExit("--raw-dir is required when --metadata is used")
        metadata = pd.read_csv(args.metadata)
        features, audit = build_feature_table(
            metadata,
            args.raw_dir,
            time_column=args.time_column,
            channel_map=channel_map,
            config=config,
            require_tu=True,
        )
        audit.to_csv(args.output_dir / "sample_accounting.csv", index=False)
    else:
        features = pd.read_csv(args.features)
        missing = [
            column
            for column in ["sample_id", "tu", *feature_columns()]
            if column not in features.columns
        ]
        if missing:
            raise SystemExit(f"precomputed feature table is missing columns: {missing}")

    features.to_csv(args.output_dir / "features_signal_quality_screened.csv", index=False)
    results = evaluate_workflow(features)
    results["screening"].to_csv(
        args.output_dir / "isolation_forest_screening.csv", index=False
    )
    results["main_predictions"].to_csv(
        args.output_dir / "loocv_predictions_main.csv", index=False
    )
    results["all_predictions"].to_csv(
        args.output_dir / "loocv_predictions_no_isolation.csv", index=False
    )
    results["ablation_metrics"].to_csv(
        args.output_dir / "ablation_metrics.csv", index=False
    )
    results["feature_importance"].to_csv(
        args.output_dir / "feature_importance.csv", index=False
    )
    results["system_importance"].to_csv(
        args.output_dir / "system_importance.csv", index=False
    )

    metrics = {
        **results["metrics"],
        "settings": {
            "random_state": 42,
            "isolation_forest_contamination": 0.04,
            "features": feature_columns(),
            "feature_extraction": asdict(config),
            "channel_map": channel_map,
            "time_column": args.time_column,
            "software": {
                "python": platform.python_version(),
                "joblib": version("joblib"),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit-learn": version("scikit-learn"),
                "scipy": version("scipy"),
            },
        },
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    model_bundle = {
        "schema_version": 1,
        "regressor": results["final_model"],
        "feature_names": feature_columns(),
        "feature_config": asdict(config),
        "channel_map": channel_map,
        "time_column": args.time_column,
    }
    joblib.dump(model_bundle, args.output_dir / "toxicity_model.joblib")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

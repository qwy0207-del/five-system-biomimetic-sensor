"""Reproducible feature extraction and regression for five-system OCP data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    IsolationForest,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


CHANNELS: tuple[str, ...] = (
    "neural",
    "metabolic",
    "circulatory",
    "immune",
    "developmental",
)
DESCRIPTORS: tuple[str, ...] = ("LDV", "KSLP", "BFGP")
RANDOM_STATE = 42


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-extraction settings reported in the manuscript."""

    baseline_start_s: float = 5.0
    baseline_end_s: float = 64.0
    reaction_start_s: float = 65.0
    reaction_end_s: float = 85.0
    sampling_interval_s: float = 0.1
    sampling_tolerance_s: float = 0.02
    savgol_window: int = 15
    savgol_polyorder: int = 3
    steady_state_samples: int = 10
    denominator_epsilon: float = 1e-6


def feature_columns(channels: Sequence[str] = CHANNELS) -> list[str]:
    """Return the locked, channel-major order of the 15 model features."""

    return [f"{channel}_{descriptor}" for channel in channels for descriptor in DESCRIPTORS]


def _validate_signal_table(
    frame: pd.DataFrame,
    time_column: str,
    channel_map: Mapping[str, str],
    config: FeatureConfig,
) -> pd.DataFrame:
    required = [time_column, *channel_map.values()]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    work = frame.loc[:, required].copy()
    for column in required:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work.isna().any().any():
        bad = work.columns[work.isna().any()].tolist()
        raise ValueError(f"non-numeric or missing values in columns: {bad}")

    work = work.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    time = work[time_column].to_numpy(dtype=float)
    if len(time) < config.savgol_window:
        raise ValueError("recording is shorter than the Savitzky-Golay window")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time values must be strictly increasing")

    relevant = (time >= config.baseline_start_s) & (time <= config.reaction_end_s)
    relevant_deltas = np.diff(time[relevant])
    if relevant_deltas.size == 0:
        raise ValueError("recording does not cover the analysis interval")
    median_delta = float(np.median(relevant_deltas))
    if not np.isclose(
        median_delta,
        config.sampling_interval_s,
        atol=config.sampling_tolerance_s,
        rtol=0.0,
    ):
        raise ValueError(
            f"median sampling interval is {median_delta:.6g} s; "
            f"expected {config.sampling_interval_s:.6g} s"
        )
    if np.any(
        relevant_deltas
        > config.sampling_interval_s + config.sampling_tolerance_s
    ):
        raise ValueError("recording contains a gap in the analysis interval")
    return work


def extract_features(
    frame: pd.DataFrame,
    *,
    time_column: str = "time_s",
    channel_map: Mapping[str, str] | None = None,
    config: FeatureConfig = FeatureConfig(),
) -> dict[str, float]:
    """Extract LDV, KSLP and BFGP from each of the five OCP channels.

    LDV is the log-transformed absolute steady-state displacement, KSLP is
    the log-transformed maximum absolute reaction-window slope, and BFGP is
    the channel baseline divided by the mean baseline across all channels.
    """

    channel_map = dict(channel_map or {channel: channel for channel in CHANNELS})
    if set(channel_map) != set(CHANNELS):
        raise ValueError(f"channel_map keys must be {CHANNELS}")
    if config.savgol_window % 2 != 1:
        raise ValueError("savgol_window must be odd")
    if config.savgol_polyorder >= config.savgol_window:
        raise ValueError("savgol_polyorder must be smaller than savgol_window")

    work = _validate_signal_table(frame, time_column, channel_map, config)
    time = work[time_column].to_numpy(dtype=float)
    baseline_mask = (time >= config.baseline_start_s) & (time <= config.baseline_end_s)
    reaction_mask = (time >= config.reaction_start_s) & (time <= config.reaction_end_s)
    if int(baseline_mask.sum()) == 0:
        raise ValueError("no observations fall in the baseline window")
    if int(reaction_mask.sum()) < max(2, config.steady_state_samples):
        raise ValueError("too few observations fall in the reaction window")

    smoothed: dict[str, np.ndarray] = {}
    baseline_means: dict[str, float] = {}
    for canonical, source_column in channel_map.items():
        values = work[source_column].to_numpy(dtype=float)
        filtered = savgol_filter(
            values,
            window_length=config.savgol_window,
            polyorder=config.savgol_polyorder,
            mode="interp",
        )
        smoothed[canonical] = filtered
        baseline_means[canonical] = float(np.mean(filtered[baseline_mask]))

    five_channel_baseline = float(np.mean(list(baseline_means.values())))
    denominator = five_channel_baseline + config.denominator_epsilon
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError("five-channel mean baseline gives an invalid BFGP denominator")

    features: dict[str, float] = {}
    for channel in CHANNELS:
        reaction_values = smoothed[channel][reaction_mask]
        v0 = baseline_means[channel]
        vss = float(np.mean(reaction_values[-config.steady_state_samples :]))
        max_slope = float(
            np.max(np.abs(np.gradient(reaction_values, config.sampling_interval_s)))
        )
        features[f"{channel}_LDV"] = float(np.log1p(abs(vss - v0)))
        features[f"{channel}_KSLP"] = float(np.log1p(max_slope))
        features[f"{channel}_BFGP"] = float(v0 / denominator)

    if not np.isfinite(np.fromiter(features.values(), dtype=float)).all():
        raise ValueError("feature extraction produced a non-finite value")
    return features


def _as_include_flag(value: object) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1", "include", "included"}:
        return True
    if normalized in {"false", "f", "no", "n", "0", "exclude", "excluded"}:
        return False
    raise ValueError(f"cannot interpret include flag: {value!r}")


def build_feature_table(
    metadata: pd.DataFrame,
    raw_dir: str | Path,
    *,
    time_column: str = "time_s",
    channel_map: Mapping[str, str] | None = None,
    config: FeatureConfig = FeatureConfig(),
    require_tu: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a feature table and a complete sample-accounting table.

    Metadata must contain ``sample_id``. ``csv_file`` defaults to
    ``<sample_id>.csv`` when omitted. ``include`` is the locked manual
    signal-quality decision; omitted values default to inclusion.
    """

    required = ["sample_id", *(("tu",) if require_tu else ())]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")
    if metadata["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id values must be unique")

    raw_dir = Path(raw_dir)
    channel_map = dict(channel_map or {channel: channel for channel in CHANNELS})
    records: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    for row in metadata.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        csv_value = row.get("csv_file")
        csv_name = f"{sample_id}.csv" if pd.isna(csv_value) else str(csv_value)
        path = raw_dir / csv_name
        include = _as_include_flag(row.get("include", True))

        if not path.is_file():
            audit.append(
                {
                    "sample_id": sample_id,
                    "csv_file": csv_name,
                    "status": "excluded",
                    "reason": "missing_raw_file",
                }
            )
            continue
        if not include:
            audit.append(
                {
                    "sample_id": sample_id,
                    "csv_file": csv_name,
                    "status": "excluded",
                    "reason": str(row.get("exclusion_reason", "manual_signal_quality")),
                }
            )
            continue

        try:
            frame = pd.read_csv(path)
            sample_features = extract_features(
                frame,
                time_column=time_column,
                channel_map=channel_map,
                config=config,
            )
        except Exception as exc:
            raise ValueError(
                f"included sample {sample_id!r} failed signal validation: {exc}. "
                "Correct the raw file or mark the sample as excluded in metadata."
            ) from exc

        record: dict[str, object] = {"sample_id": sample_id}
        if "tu" in row:
            record["tu"] = row["tu"]
        record.update(sample_features)
        records.append(record)
        audit.append(
            {
                "sample_id": sample_id,
                "csv_file": csv_name,
                "status": "included",
                "reason": "passed_manual_and_structural_checks",
            }
        )

    features = pd.DataFrame.from_records(records)
    if features.empty:
        raise ValueError("no samples were available for analysis")
    if require_tu:
        features["tu"] = pd.to_numeric(features["tu"], errors="coerce")
        if features["tu"].isna().any() or (features["tu"] < 0).any():
            raise ValueError("TU labels must be finite, numeric and non-negative")
    return features, pd.DataFrame.from_records(audit)


def make_regressor() -> Pipeline:
    """Construct the locked scaler and weighted ensemble regressor."""

    extra_trees = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    gradient_boosting = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.06,
        max_depth=5,
        random_state=RANDOM_STATE,
    )
    ensemble = VotingRegressor(
        estimators=[("extra_trees", extra_trees), ("gradient_boosting", gradient_boosting)],
        weights=[0.4, 0.6],
        n_jobs=1,
    )
    return Pipeline([("scaler", RobustScaler()), ("regressor", ensemble)])


def _validate_feature_matrix(
    table: pd.DataFrame, columns: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    missing = [column for column in ["tu", *columns] if column not in table.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")
    x = table.loc[:, columns].to_numpy(dtype=float)
    y = table["tu"].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and TU labels must all be finite")
    if len(y) < 3:
        raise ValueError("at least three samples are required")
    return x, y


def isolation_screen(
    table: pd.DataFrame, columns: Sequence[str], contamination: float = 0.04
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the manuscript's once-only, label-free pre-screening step."""

    x, _ = _validate_feature_matrix(table, columns)
    detector = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    keep = detector.fit_predict(x) == 1
    return keep, detector.decision_function(x)


def loocv_predictions(
    table: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Generate clipped, out-of-fold predictions with fold-local scaling."""

    x, y = _validate_feature_matrix(table, columns)
    predicted = np.empty(len(table), dtype=float)
    splitter = LeaveOneOut()
    for train_index, test_index in splitter.split(x):
        model = make_regressor()
        model.fit(x[train_index], y[train_index])
        predicted[test_index] = np.maximum(model.predict(x[test_index]), 0.0)
    return pd.DataFrame(
        {
            "sample_id": table["sample_id"].astype(str).to_numpy(),
            "tu_true": y,
            "tu_predicted": predicted,
            "absolute_error": np.abs(predicted - y),
        }
    )


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Calculate the regression and operational-agreement metrics."""

    y_true = predictions["tu_true"].to_numpy(dtype=float)
    y_pred = predictions["tu_predicted"].to_numpy(dtype=float)
    strict = np.abs(y_pred - y_true) <= 0.2
    high_tu_interval = (y_true >= 0.8) & (y_pred >= 0.6)
    pearson_r = float(pearsonr(y_true, y_pred).statistic)
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": pearson_r,
        "strict_agreement": float(np.mean(strict)),
        "strict_or_high_tu_interval_agreement": float(np.mean(strict | high_tu_interval)),
    }


def _without_fusion_table(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Collapse channels by descriptor, removing channel-level feature fusion."""

    reduced = table.loc[:, ["sample_id", "tu"]].copy()
    columns: list[str] = []
    for descriptor in DESCRIPTORS:
        name = f"mean_{descriptor}"
        source = [f"{channel}_{descriptor}" for channel in CHANNELS]
        reduced[name] = table.loc[:, source].mean(axis=1)
        columns.append(name)
    return reduced, columns


def _weighted_feature_importance(
    fitted_model: Pipeline, columns: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    voter = fitted_model.named_steps["regressor"]
    extra_trees = voter.named_estimators_["extra_trees"]
    gradient_boosting = voter.named_estimators_["gradient_boosting"]
    weights = np.asarray(voter.weights, dtype=float)
    weights /= weights.sum()
    importance = (
        weights[0] * extra_trees.feature_importances_
        + weights[1] * gradient_boosting.feature_importances_
    )
    feature_importance = pd.DataFrame(
        {"feature": list(columns), "relative_importance": importance}
    ).sort_values("relative_importance", ascending=False, ignore_index=True)

    rows = []
    for channel in CHANNELS:
        value = feature_importance.loc[
            feature_importance["feature"].str.startswith(f"{channel}_"),
            "relative_importance",
        ].sum()
        rows.append({"system": channel, "relative_importance": float(value)})
    system_importance = pd.DataFrame(rows).sort_values(
        "relative_importance", ascending=False, ignore_index=True
    )
    return feature_importance, system_importance


def fit_final_model(
    table: pd.DataFrame, columns: Sequence[str] | None = None
) -> Pipeline:
    """Fit the locked model to all rows supplied in ``table``."""

    columns = list(columns or feature_columns())
    x, y = _validate_feature_matrix(table, columns)
    model = make_regressor()
    model.fit(x, y)
    return model


def evaluate_workflow(table: pd.DataFrame) -> dict[str, object]:
    """Run the main analysis, sensitivity analysis, ablations and importance."""

    columns = feature_columns()
    keep, scores = isolation_screen(table, columns, contamination=0.04)
    screening = table.loc[:, ["sample_id", "tu"]].copy()
    screening["isolation_score"] = scores
    screening["retained"] = keep

    main_table = table.loc[keep].reset_index(drop=True)
    main_predictions = loocv_predictions(main_table, columns)
    all_predictions = loocv_predictions(table.reset_index(drop=True), columns)

    ablations: list[dict[str, float | int | str]] = []
    no_temporal = [column for column in columns if not column.endswith("_KSLP")]
    no_temporal_predictions = loocv_predictions(main_table, no_temporal)
    ablations.append(
        {"analysis": "no_temporal_features", **regression_metrics(no_temporal_predictions)}
    )

    no_fusion_table, no_fusion_columns = _without_fusion_table(main_table)
    no_fusion_predictions = loocv_predictions(no_fusion_table, no_fusion_columns)
    ablations.append({"analysis": "no_fusion", **regression_metrics(no_fusion_predictions)})

    for channel in CHANNELS:
        channel_columns = [f"{channel}_{descriptor}" for descriptor in DESCRIPTORS]
        predictions = loocv_predictions(main_table, channel_columns)
        ablations.append(
            {"analysis": f"{channel}_only", **regression_metrics(predictions)}
        )

    final_model = fit_final_model(main_table, columns)
    feature_importance, system_importance = _weighted_feature_importance(
        final_model, columns
    )
    return {
        "screening": screening,
        "main_table": main_table,
        "main_predictions": main_predictions,
        "all_predictions": all_predictions,
        "metrics": {
            "main_isolation_filtered": regression_metrics(main_predictions),
            "sensitivity_no_isolation_filter": regression_metrics(all_predictions),
        },
        "ablation_metrics": pd.DataFrame.from_records(ablations),
        "feature_importance": feature_importance,
        "system_importance": system_importance,
        "final_model": final_model,
        "feature_config": asdict(FeatureConfig()),
    }

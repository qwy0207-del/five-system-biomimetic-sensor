# Five-system biomimetic sensor machine-learning code

This repository contains only the machine-learning workflow used for toxicity-unit (TU) prediction in *Five-system biomimetic sensing for early warning of waterborne health risks*. It does not contain plotting, manuscript-generation, hardware-control or finite-element simulation code.

## Included files

- `src/toxicity_ml.py`: OCP feature extraction, label-free Isolation Forest screening, leave-one-out cross-validation (LOOCV), ensemble regression, ablation analyses and feature importance.
- `run_analysis.py`: command-line entry point for training and evaluation.
- `predict.py`: command-line entry point for applying the fitted model to new OCP recordings.
- `requirements.txt`: pinned Python dependencies.

## Input data

The sample manifest is a CSV containing:

- `sample_id`: unique sample identifier.
- `tu`: non-negative ground-truth TU label (required for training only).
- `csv_file`: raw recording filename; if omitted, `<sample_id>.csv` is used.
- `include`: optional locked manual signal-quality decision (`true` or `false`).
- `exclusion_reason`: optional reason recorded when `include` is false.

Each raw CSV must contain a time column and five synchronized OCP columns. The default names are:

```text
time_s,neural,metabolic,circulatory,immune,developmental
```

Recordings must retain the same voltage unit across all samples. The code does not silently impute, resample or change units. Files marked for inclusion must contain numeric data covering the 5-64 s baseline window and 65-85 s reaction window at a 0.1 s sampling interval.

## Reproduce the analysis

Create a Python 3.11 or newer environment and install the pinned, tested packages:

```bash
python -m pip install -r requirements.txt
```

Run the complete workflow from raw recordings:

```bash
python run_analysis.py \
  --metadata data/sample_manifest.csv \
  --raw-dir data/raw \
  --output-dir results
```

If the raw CSV columns have different names, provide them in the fixed order neural, metabolic, circulatory, immune and developmental:

```bash
python run_analysis.py \
  --metadata data/sample_manifest.csv \
  --raw-dir data/raw \
  --output-dir results \
  --time-column Time \
  --channel-columns Channel1 Channel2 Channel3 Channel4 Channel5
```

A locked precomputed feature table may be analysed directly:

```bash
python run_analysis.py \
  --features data/features_77.csv \
  --output-dir results
```

The feature table must contain `sample_id`, `tu` and the 15 channel-major feature columns named `<system>_<descriptor>`, where the systems are `neural`, `metabolic`, `circulatory`, `immune` and `developmental`, and the descriptors are `LDV`, `KSLP` and `BFGP`.

## Locked workflow

For each channel, the raw trace is smoothed using a Savitzky-Golay filter (window length 15, polynomial order 3). The baseline mean is calculated over 5-64 s, and the steady-state value is the mean of the final ten samples in the 65-85 s reaction window. The descriptors are:

```text
LDV_j  = ln(1 + |Vss,j - V0,j|)
KSLP_j = ln(1 + max |dVj/dt|)
BFGP_j = V0,j / (mean_j(V0,j) + 1e-6)
```

After the locked manual signal-quality decisions are applied, Isolation Forest (`contamination=0.04`, `random_state=42`) is fitted once to the feature matrix without TU labels. LOOCV is then performed on the retained fixed analysis set. A `RobustScaler` is fitted only to the training samples within each fold. The Voting Regressor combines:

- Extra Trees: 300 estimators, maximum depth 12, weight 0.4.
- Gradient Boosting: 200 estimators, learning rate 0.06, maximum depth 5, weight 0.6.

All random states are 42, and predictions are clipped to non-negative TU. The code also runs LOOCV on all signal-quality-screened samples without Isolation Forest filtering. Operational agreement uses the manuscript definitions: strict agreement is absolute error at most 0.2 TU; the additional high-TU interval rule requires true TU at least 0.8 and predicted TU at least 0.6.

The ablation analyses use the same fixed analysis set. Removing temporal features excludes the five KSLP features. Removing five-system fusion averages each descriptor across the five systems before modelling. Five additional models use the three descriptors from one sensing system at a time. No model or hyperparameter selection is performed during LOOCV.

Because Isolation Forest screening precedes fold definition, the primary metrics describe the pre-screened population rather than a fully nested estimate. The unfiltered analysis is retained as the specified sensitivity analysis.

## Outputs

`run_analysis.py` writes machine-readable CSV/JSON files only; it creates no figures. The principal outputs are the two LOOCV prediction tables, regression and operational metrics, ablation metrics, feature and system importance tables, the complete screening record and a fitted `toxicity_model.joblib` bundle.

To predict TU for new recordings, use a manifest containing `sample_id`, `csv_file` and optional `include` columns:

```bash
python predict.py \
  --model results/toxicity_model.joblib \
  --metadata data/new_samples.csv \
  --raw-dir data/new_raw \
  --output results/new_predictions.csv
```

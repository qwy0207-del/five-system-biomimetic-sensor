# Five-system biomimetic sensor machine-learning code

This repository provides the machine-learning workflow for toxicity-unit (TU) prediction associated with *Five-system biomimetic sensing for early warning of waterborne health risks*, including OCP feature extraction, screening, model evaluation, ablation analyses and prediction on new recordings.

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

Install the pinned dependency set for the analysis workflow:

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

Isolation Forest screening precedes fold definition. The primary analysis evaluates the fixed pre-screened population, and the complementary sensitivity analysis evaluates all signal-quality-screened samples.

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


## System requirements and tested environment

The complete bundled demo and new-recording prediction were successfully run on the following desktop configuration on 2 October 2026:

| Component | Tested configuration |
| --- | --- |
| Operating system | Windows 11 Pro 25H2, 64-bit, OS build 26200.9457 |
| Processor | Intel Core Ultra 9 275HX, 24 cores |
| Memory | 32 GB installed class (31.44 GiB reported by Windows) |
| Python | 3.13.5 |
| NumPy | 2.4.3 |
| pandas | 3.0.1 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |

No non-standard hardware is required for offline analysis of recorded OCP data. Real-time OCP acquisition uses the custom sensing hardware described in the manuscript. The configuration above identifies the tested desktop; it is not a minimum hardware specification.

## Installation guide for the tested Windows environment

Install Python 3.13.5, download or clone this repository, open PowerShell in its root directory and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-tested.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements-windows-tested.txt` pins the successfully tested desktop dependency versions. `requirements.txt` retains the existing pinned dependency set.

Dependency installation into a newly created virtual environment took 123 seconds (approximately 2 minutes) on the desktop above. This measurement includes package download and installation using the configured package index and excludes Python installation and virtual-environment creation. Download speed and package caching affect installation time.

## Bundled demo

`demo/raw/` contains 12 deterministic simulated five-channel OCP recordings, each covering 0-90 seconds at 0.1-second intervals. `demo/sample_manifest.csv` supplies illustrative TU labels spanning 0-2 TU. These simulated recordings are supplied for software demonstration; manuscript results are based on the experimental datasets described in the paper.

Run the complete demo from the repository root:

```powershell
.\.venv\Scripts\python.exe run_analysis.py --metadata demo/sample_manifest.csv --raw-dir demo/raw --output-dir demo/results
```

The measured end-to-end demo runtime was 25.4 seconds on the tested desktop. The demo extracts 12 feature sets, retains 11 after Isolation Forest screening and completes the primary and sensitivity LOOCV analyses, ablation analyses and final model fitting.

### Expected output

The command writes these files to `demo/results/`:

- `sample_accounting.csv`
- `features_signal_quality_screened.csv`
- `isolation_forest_screening.csv`
- `loocv_predictions_main.csv`
- `loocv_predictions_no_isolation.csv`
- `ablation_metrics.csv`
- `feature_importance.csv`
- `system_importance.csv`
- `metrics.json`
- `toxicity_model.joblib`

Reference CSV/JSON outputs from the tested demo are supplied in `demo/expected_output/`. These files provide software verification targets for the simulated demo. Numerical results can vary slightly between computing environments.

### Predict on additional recordings

After completing the demo, run:

```powershell
.\.venv\Scripts\python.exe predict.py --model demo/results/toxicity_model.joblib --metadata demo/prediction_manifest.csv --raw-dir demo/raw --output demo/results/new_predictions.csv
```

This example produces predictions for three included recordings and a corresponding sample-accounting file. Its measured end-to-end runtime was 4.1 seconds on the tested desktop.

## Instructions for use with experimental data

Prepare a manifest and synchronized raw CSV recordings using the input schema above. Maintain consistent voltage units and channel order across training and prediction. Recordings should cover the specified baseline and reaction windows. The `include` and `exclusion_reason` fields support a documented signal-quality decision for each recording.

Run `run_analysis.py` with the experimental manifest and raw-data directory to obtain features, screening records, cross-validation predictions and a fitted model. Run `predict.py` with that model and a new-sample manifest to obtain predicted TU values. The sample IDs and labels in the supplied demo illustrate the required schema; use the experimental sample accounting and TU labels for manuscript-result reproduction.

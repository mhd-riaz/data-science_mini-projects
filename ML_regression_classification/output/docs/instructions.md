# Reproduction instructions

**Enrollment ID:** `PES1PGE25DS037` — ML1 Hackathon, Bengaluru house price regression.

Every step below is executed from the **repository root**, i.e. the directory that contains
`ML_regression_classification/`. Nothing in `ML_regression_classification/data/` is modified.

---

## Step 1 — Verify the input data is present

```bash
ls ML_regression_classification/data/
# expected: avg_rent.csv  dist_from_city_centre.csv  sample_submission.csv  test.csv  train.csv
```

Expected shapes: `train.csv` 10,656 × 10, `test.csv` 2,664 × 9, `sample_submission.csv` 2,664 × 2,
`avg_rent.csv` 157 × 2, `dist_from_city_centre.csv` 500 × 2.

## Step 2 — Create the environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r ML_regression_classification/output/requirements.txt
```

Confirm the versions the results were produced with:

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; \
print(pandas.__version__, numpy.__version__, sklearn.__version__, matplotlib.__version__, seaborn.__version__)"
# 3.0.5 2.5.1 1.9.0 3.11.1 0.13.2
```

## Step 3 — Run the pipeline

Non-interactive (recommended, single command):

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  ML_regression_classification/output/notebook/PES1PGE25DS037_hackathon.ipynb
```

Interactive alternative: open the same notebook in Jupyter Lab or VS Code and select **Run All**.
No cell requires editing — paths are resolved automatically by walking up from the working
directory until `data/train.csv` is found, and `RANDOM_STATE = 42` is fixed throughout.

Runtime: approximately 3–4 minutes (the 16-configuration model comparison and the 5-fold
cross-validation dominate it).

## Step 4 — Collect the outputs

| Artefact | Path |
|---|---|
| Submission CSV | `ML_regression_classification/output/submission/PES1PGE25DS037_submission.csv` |
| Figures | `ML_regression_classification/output/report/figures/01..07_*.png` |
| Executed notebook (with outputs) | `ML_regression_classification/output/notebook/PES1PGE25DS037_hackathon.ipynb` |

## Step 5 — Verify the submission

The notebook already asserts the submission contract in the final cell of Section 8; it prints five
`PASS` lines and raises `AssertionError` if any check fails. To re-verify independently:

```bash
python - <<'CHECK'
import pandas as pd
base = "ML_regression_classification/"
sub = pd.read_csv(base + "output/submission/PES1PGE25DS037_submission.csv")
ref = pd.read_csv(base + "data/sample_submission.csv")
test = pd.read_csv(base + "data/test.csv")
assert list(sub.columns) == list(ref.columns)
assert sub.dtypes.to_dict() == ref.dtypes.to_dict()
assert len(sub) == len(test) == len(ref)
assert (sub["ID"].values == test["ID"].values).all()
assert sub["price"].notna().all() and (sub["price"] > 0).all()
print("submission OK:", sub.shape, list(sub.columns))
CHECK
```

## Step 6 — Confirm the reported metric

Section 7 of the notebook prints the headline block:

```
FINAL MODEL - held-out validation (20%, n=2132, never cleaned)
  RMSE : 80.258 Lakhs
  MAE  : 26.137 Lakhs
  R^2  : 0.7491
  5-fold CV RMSE : 72.747 +/- 6.875
```

These are the values quoted in `report/PES1PGE25DS037_report.md`. They are deterministic for the
pinned package versions and `RANDOM_STATE = 42`; small deviations (< 1 Lakh) can occur on different
scikit-learn or BLAS builds because of floating-point reduction order in the ensembles.

---

## Notebook section map

| Section | What it does |
|---|---|
| 1 | Imports, path resolution, reads the submission contract from `sample_submission.csv` |
| 2 | EDA: missing values, target distribution, messy-field audit, cardinality |
| 3 | Feature engineering: `total_sqft`/`size`/`availability` parsing, fuzzy joins to both external datasets, derived ratios |
| 4 | Preprocessing pipeline: median imputation, one-hot, target encoding, scaling |
| 5 | 80/20 stratified split; outlier-policy and external-data ablations |
| 6 | 8 models × 2 target scales compared on the untouched holdout; blend-weight sweep |
| 7 | 5-fold cross-validation of the finalists; final metrics; residual and feature-importance plots |
| 8 | Refit on all clean training rows, predict `test.csv`, write and assert the submission |
| 9 | Conclusions and limitations |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: Could not locate .../data/train.csv` | Notebook launched outside the repository tree — `cd` to the repository root and rerun. |
| `nbconvert` times out | Increase `--ExecutePreprocessor.timeout`. |
| Slightly different RMSE | Different scikit-learn/BLAS build; verify the versions from Step 2. |
| Missing figures | `output/report/figures/` is created by the notebook; rerun Step 3 rather than creating it manually. |

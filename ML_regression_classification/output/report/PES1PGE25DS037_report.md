# ML1 Hackathon Report — Predicting Bengaluru House Prices

**Student:** Mohamed Riaz  **Enrollment ID:** `PES1PGE25DS037`
**Programme:** M.Tech (Data Science & Machine Learning), PES College
**Course:** ML1 — Supervised Machine Learning: Regression (SMLR) and Classification (SMLC)
**Deliverable notebook:** `output/notebook/PES1PGE25DS037_hackathon.ipynb`

---

## 1. Problem framing

Predict the sale `price` (in **Lakhs**) of a Bengaluru residential listing from nine descriptive
attributes, so a buyer or renter can judge whether an asking price is fair. This is a **supervised
regression** problem in the sense of *SMLR Session 1 — Types of Machine Learning*: a continuous,
strictly positive target with a labelled training set of 10,656 rows, scored on a held-out set of
2,664 rows by **RMSE**.

Two properties of the target drive most design decisions:

| Statistic | Value |
|---|---|
| Rows (train / test) | 10,656 / 2,664 |
| price: min / median / mean / max (Lakhs) | 8 / 72 / 112.2 / 2,700 |
| price: standard deviation | 142.2 |
| Skewness | 6.75 |
| 99th percentile / maximum | 660 / 2,700 |

The distribution is extremely right-skewed (Figure `01_target_distribution.png`). Because RMSE
squares the error, the ~1 % of listings above 660 Lakhs contribute a disproportionate share of the
loss. **The metric, not the bulk of the data, dictates the modelling choices** — this is the single
most consequential observation in this report and it is revisited in Sections 4 and 6.

Two additional reference tables are supplied for feature engineering: `avg_rent.csv` (average 2BHK
rent for 157 localities) and `dist_from_city_centre.csv` (distance from the city centre for 500
localities).

---

## 2. Exploratory data analysis

### 2.1 Missing values

*SMLR Session 1 — Missing Values (Standard and Non-Standard)*:

| Column | Missing in train | % | Missing in test |
|---|---|---|---|
| `society` | 4,428 | 41.6 % | 1,074 |
| `balcony` | 504 | 4.7 % | 105 |
| `bath` | 65 | 0.6 % | 8 |
| `size` | 14 | 0.1 % | 2 |
| `location` | 1 | 0.01 % | 0 |

`society` missingness is not random — unnamed properties are predominantly independent houses and
plots rather than branded apartment projects. It is therefore treated as **informative
missingness**: an explicit `society_missing` indicator is created rather than the rows being
discarded. The same indicator treatment is applied to `bath` and `balcony`.

### 2.2 Non-standard missing values and messy encodings

Three columns are free text rather than clean types:

| Column | Problem | Frequency |
|---|---|---|
| `total_sqft` | Ranges (`"2100 - 2850"`) and seven non-sqft units (`Sq Meter`, `Sq Yards`, `Perch`, `Acres`, `Cents`, `Guntha`, `Grounds`) | 206 train rows (1.9 %), 41 test rows |
| `size` | Mixed vocabulary: `"2 BHK"` (8,586), `"4 Bedroom"` (2,043), `"1 RK"` (13) | all rows |
| `availability` | 79 distinct values: `"Ready To Move"` (79.5 %), `"Immediate Possession"`, and `"YY-Mon"` future-possession codes | all rows |

These are exactly the *non-standard missing values* of *SMLR Session 1*: naively coercing
`total_sqft` with `pd.to_numeric` silently produces 206 `NaN`s, discarding real information —
including a `5.31Acres` plot that is genuinely one of the largest properties in the dataset.

### 2.3 Cardinality

`location` has 1,197 distinct values and `society` 2,364, against 10,656 rows; 809 locations appear
in fewer than five listings. One-hot encoding these two columns alone would create ~3,561 sparse
binary columns — more columns than rows, guaranteeing overfitting. This directly motivates the
encoding choice in Section 3.4.

### 2.4 Relationship with the target

Correlation with `price` (*SMLR Session 2 — Covariance & Correlation*):

| Feature | Pearson | Spearman |
|---|---|---|
| `sqft` (parsed) | 0.047 | **0.734** |
| `bath` | 0.460 | 0.717 |
| `bhk` | 0.406 | 0.689 |
| `dist_from_city` (external) | −0.197 | −0.286 |
| `balcony` | 0.126 | 0.213 |
| `avg_2bhk_rent` (external) | 0.049 | 0.205 |
| `loc_freq` | −0.043 | −0.095 |

The near-zero **Pearson** value for `sqft` against a **Spearman of 0.734** is a diagnostic finding,
not a real absence of signal: a handful of Acre- and Guntha-denominated land parcels convert to
values four orders of magnitude above a typical flat, which destroys a linear correlation
coefficient while leaving the rank relationship intact. `sqft`/`log_sqft` end up as the two most
important features in the fitted model (Section 7). The negative sign on `dist_from_city` confirms
the expected economic gradient: prices fall with distance from the centre.

---

## 3. Feature engineering

*SMLR Session 4 — Feature Engineering* distinguishes **feature extraction**, **transformation**,
**scaling** and **selection**. All four are applied; 28 model-ready features are produced from the
nine raw columns plus the two external tables.

### 3.1 Extraction from messy text (`parse_sqft`, `parse_bhk`, `size_kind`, `parse_availability`)

| Raw value | Engineered output |
|---|---|
| `"1000 - 1200"` | `sqft = 1100` (midpoint) |
| `"34.46Sq. Meter"` | `sqft = 370.9` (× 10.7639) |
| `"5.31Acres"` | `sqft = 231,303.6` (× 43,560) |
| `"4 Bedroom"` | `bhk = 4`, `size_kind = "Bedroom"` |
| `"1 RK"` | `bhk = 1`, `size_kind = "RK"` |
| `"Ready To Move"` | `avail_ready = 1`, `avail_months = 0` |
| `"19-Dec"` | `avail_months = 23` (months after Jan-2018) |

The **suffix of `size` is retained as its own categorical**, not discarded: `"Bedroom"` listings are
overwhelmingly plots and villas, `"BHK"` listings are apartments, and the two have materially
different price-per-square-foot regimes. Discarding the suffix would silently merge two populations.

### 3.2 Joining the external datasets (the core feature-engineering task)

Both reference tables key on `location`, but spellings disagree with `train.csv`
(`"Sarjapur  Road"` vs `"Sarjapur Road"`, `"Whitefield Hope Farm Junction"` vs `"Whitefield"`). A
plain exact join is inadequate. A three-stage matcher is used on a normalised key (lower-cased,
punctuation stripped, whitespace collapsed):

1. exact match on the normalised key;
2. longest **substring containment** in either direction (≥ 5 characters, to avoid spurious hits);
3. `difflib` sequence similarity ≥ 0.87.

Measured coverage:

| Reference table | Exact join (train / test) | Three-stage matcher (train / test) |
|---|---|---|
| `avg_rent.csv` | 38.1 % / 38.9 % | **58.4 % / 59.2 %** |
| `dist_from_city_centre.csv` | 90.6 % / 89.9 % | **93.3 % / 93.0 %** |

Fuzzy matching recovers a further 20 percentage points of rent coverage — roughly 2,150 additional
training rows that would otherwise have had no rent signal at all. Rows that still fail to match
receive median imputation **plus** an explicit `rent_missing` / `dist_missing` flag, so the model
can learn that "unmatched" is itself informative (unmatched localities are typically peripheral).

### 3.3 Derived ratios and interactions

*SMLR Session 3 — Interaction effect* and *Session 4 — Feature Transformation*:

| Feature | Rationale |
|---|---|
| `sqft_per_bhk`, `bath_per_bhk`, `balcony_per_bhk` | Normalise size against room count — separates spacious 2BHKs from cramped 4BHKs |
| `log_sqft` | Linearises the multiplicative area–price relationship |
| `rent_per_sqft` = rent / sqft | Locality rental yield proxy |
| `rent_x_sqft`, `dist_x_sqft` | Explicit interaction terms: locality price level scaled by property size |
| `inv_dist` = 1/(1 + distance) | Captures the sharply non-linear premium of the innermost localities |
| `loc_freq`, `soc_freq` | Listing-count popularity; computed from train **and** test *features only*, never the target, so no leakage |

### 3.4 Encoding (*SMLR Session 1 — Handle Non-Numeric Data*)

| Block | Columns | Encoder | Why |
|---|---|---|---|
| Low-cardinality categorical | `area_type` (4), `size_kind` (4) | `OneHotEncoder` | Few, unordered levels — the textbook one-hot case |
| High-cardinality categorical | `location` (1,197), `society` (2,364) | `TargetEncoder` (internally cross-fitted) | One-hot would add ~3,561 columns to 10,656 rows |
| Numeric | 24 features | `SimpleImputer(median)` | Median resists the heavy right tail; mean would be dragged by outliers |
| Scaling | numerics, linear models only | `StandardScaler` | *SMLR Session 4 — Feature Scaling*; required by Ridge/Lasso, irrelevant to trees |

Ordinal encoding was rejected for `location`: locality names have no natural order, and imposing an
arbitrary integer ordering would inject a false monotonic relationship. Scikit-learn's
`TargetEncoder` was chosen over a hand-rolled group mean specifically because it cross-fits
internally, which prevents the target leakage that naive target encoding is notorious for.

Every transformer is fitted **inside** a `Pipeline`/`ColumnTransformer` so that all statistics
(medians, category means, scaler parameters) are learned from the training fold only —
*SMLR Session 6 — Create ML pipeline*.

---

## 4. Outlier handling — a textbook rule, tested rather than assumed

*SMLR Session 1 — Outliers Detection/removal (Boxplot, IQR, Z-score)* prescribes IQR trimming, and
the standard treatment of this dataset trims price-per-square-foot outliers. That rule was
**evaluated empirically on an untouched holdout instead of being assumed**, with the probe model
held fixed (Random Forest, raw target):

| Training-set policy | n_train | Validation RMSE |
|---|---|---|
| No cleaning | 8,524 | 83.684 |
| **Structural cleaning only (adopted)** | 8,296 | **82.612** |
| Structural + IQR (k = 3) on price/sqft | 8,023 | 109.031 |
| Structural + IQR (k = 1.5) on price/sqft | 7,645 | 120.425 |

**IQR trimming makes the model 46 % worse.** The reason is the interaction between the outlier rule
and the scoring metric: the "outliers" it deletes are precisely the genuinely expensive properties
that dominate a squared-error loss. Trained without them, the model has no basis for predicting
high prices and systematically under-predicts the very rows that determine RMSE. Only
*structurally implausible* records are therefore removed — 281 rows (2.64 %) with unparseable or
zero area, under 200 sqft per room, or more bathrooms than rooms + 4.

A second methodological point matters as much as the rule itself: **cleaning is applied to training
rows only.** An earlier iteration of this analysis cleaned the validation set too and reported
RMSE ≈ 49 — an artefact, because the competition test set cannot be cleaned. Every number in this
report is measured against an untouched holdout that retains its full heavy tail.

---

## 5. Validation protocol

*SMLR Session 1 — Train Test Split* and *Session 5 — Model Validation*:

- **80/20 split stratified on price deciles** (`random_state = 42`), so the rare expensive listings
  are represented proportionally in both parts. Holdout = 2,132 rows, left completely untouched.
- **5-fold cross-validation** for the finalists, with structural cleaning re-applied inside each
  fold to the training portion only.
- All preprocessing inside the pipeline, so no fold ever sees statistics derived from its own
  validation rows.

---

## 6. Model comparison

Candidates span the regularised linear family (*SMLR Session 5 — Regularization: Lasso, Ridge*) and
the ensemble family (*SMLC Unit 4 — Ensemble Learning: Random Forest, Feature Importance, Bagging,
Boosting Algorithms: AdaBoost, Gradient Boosting*). Each was fitted on the raw target and on
`log1p(price)` with **Duan's smearing** retransformation (the mean of the exponentiated training
residuals) to correct the downward bias of naively exponentiating a log-scale prediction.

| Rank | Model | Target | Val RMSE | Val MAE | Val R² |
|---|---|---|---|---|---|
| 1 | **RandomForest** | raw | **82.36** | 27.05 | 0.736 |
| 2 | GradientBoosting | log | 83.32 | 26.69 | 0.730 |
| 3 | ExtraTrees | raw | 84.26 | 27.89 | 0.723 |
| 4 | GradientBoosting | raw | 84.41 | 28.44 | 0.722 |
| 5 | RandomForest | log | 85.40 | 26.43 | 0.716 |
| 6 | HistGradientBoosting | raw | 88.28 | 28.43 | 0.696 |
| 7 | HistGradientBoosting | log | 90.03 | 26.81 | 0.684 |
| 8 | ExtraTrees | log | 91.68 | 27.40 | 0.673 |
| 9 | LinearRegression | log | 101.92 | 33.11 | 0.595 |
| 10 | AdaBoost | log | 109.33 | 37.76 | 0.534 |
| 11 | Lasso (LassoCV) | log | 110.12 | 34.02 | 0.528 |
| 12 | Ridge (RidgeCV) | log | 110.35 | 34.71 | 0.526 |
| 13 | LinearRegression | raw | 111.74 | 43.15 | 0.514 |
| 14 | Lasso (LassoCV) | raw | 112.13 | 43.37 | 0.510 |
| 15 | Ridge (RidgeCV) | raw | 112.20 | 43.33 | 0.510 |
| 16 | AdaBoost | raw | 116.24 | 77.51 | 0.474 |

Observations:

- **Ensembles beat regularised linear models by ~28 RMSE.** Ridge and Lasso (tuned by internal CV
  over 20–25 alphas) barely improve on ordinary least squares, which tells us the linear model is
  **bias-limited, not variance-limited** — regularisation cannot fix a mis-specified functional
  form. Price depends on area *interacted* with locality, a product a purely additive model cannot
  express. This is the bias/variance trade-off of *SMLR Session 5* observed directly.
- **The log transform helps boosting but hurts bagging.** Gradient Boosting improves from 84.41 to
  83.32 on the log scale (its sequential residual fitting benefits from a symmetric target), while
  Random Forest degrades from 82.36 to 85.40 — averaging log-scale leaf values then exponentiating
  understates the expensive listings that RMSE punishes most.
- **AdaBoost is the weakest ensemble**, consistent with its exponential loss being highly sensitive
  to outliers — an unfavourable property on a target with skewness 6.75.

### External-data ablation

The same probe model, identical in every respect except the feature set:

| Feature set | n_features | Validation RMSE |
|---|---|---|
| Without `avg_rent` + `dist_from_city` | 20 | 86.837 |
| **With both external datasets** | 28 | **82.612** |

The external datasets are worth **4.225 RMSE** (4.9 % relative). They are genuinely used, not
merely loaded: `dist_from_city`, `inv_dist`, `dist_x_sqft` and `rent_x_sqft` together account for
about 9.9 % of the final model's feature importance.

### Hyperparameter tuning

*SMLR Session 5 — Hyperparameter Tuning: Grid Search, Random Search.* A `RandomizedSearchCV` over
the Gradient Boosting grid (20 draws × 3-fold CV, 384 candidate combinations in the space) selected
`n_estimators=900, learning_rate=0.02, max_depth=5, min_samples_leaf=10, subsample=0.9,
max_features=0.6` — which scored **84.234** on the holdout against **83.322** for the hand-set
configuration. The search optimises mean CV error on the cleaned training folds, whereas the
holdout retains its full heavy tail, so the more heavily regularised winner generalises slightly
worse to the extreme listings. The simpler hand-set configuration was therefore kept, and the
search is retained in the notebook as evidence rather than removed.

### Blending

The two best models come from different families and make partly uncorrelated errors — the
bagging-vs-boosting distinction of *SMLC Unit 4*. A weight sweep confirms a broad, flat optimum
around equal weights:

| Weight on RandomForest | 0.0 | 0.2 | 0.4 | **0.5** | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| Validation RMSE | 83.32 | 81.49 | 80.46 | **80.26** | 80.26 | 80.90 | 82.36 |

Equal weights were **fixed a priori** rather than tuned on the holdout, to keep the reported figure
an honest generalisation estimate rather than a selection-biased one.

---

## 7. Final model and results

**Final model:** 50 % `RandomForestRegressor` (600 trees, `max_features = 0.5`, raw target) +
50 % `GradientBoostingRegressor` (600 stages, `learning_rate = 0.05`, `max_depth = 4`,
`subsample = 0.9`, `log1p` target with Duan smearing), over 28 engineered features.

| Metric | Value |
|---|---|
| **Held-out validation RMSE** (n = 2,132, never cleaned) | **80.258 Lakhs** |
| Held-out MAE | 26.137 Lakhs |
| Held-out R² | 0.7491 |
| 5-fold CV RMSE (blend) | **72.747 ± 6.875** |
| 5-fold CV RMSE (RandomForest alone) | 74.884 ± 6.000 |
| 5-fold CV RMSE (GradientBoosting-log alone) | 74.943 ± 8.152 |

The blend has the lowest mean CV RMSE of the three, and its fold-to-fold spread sits between the
two components — the variance-reduction argument for ensembling, confirmed on real folds rather
than asserted. The holdout figure (80.26) is above the CV mean (72.75) because that particular
stratified fifth happens to contain a denser set of very expensive listings; the ± 6.9 CV standard
deviation shows this variability is expected on a target this heavy-tailed.

### Feature importance (*SMLC Unit 4 — Random Forest Classifier / Feature Importance*)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `log_sqft` | 0.223 |
| 2 | `sqft` | 0.223 |
| 3 | `sqft_per_bhk` | 0.109 |
| 4 | `location` (target-encoded) | 0.096 |
| 5 | `bath` | 0.050 |
| 6 | `bhk` | 0.039 |
| 7 | `society` (target-encoded) | 0.039 |
| 8 | `inv_dist` *(external)* | 0.032 |
| 9 | `dist_from_city` *(external)* | 0.030 |
| 10 | `dist_x_sqft` *(external)* | 0.025 |

Area and locality dominate, exactly as domain intuition predicts — a useful sanity check that the
pipeline has not learned an artefact. The engineered `sqft_per_bhk` ratio outranks `bhk` itself,
justifying the derived-ratio work of Section 3.3.

### Residual diagnostics (*SMLR Session 3 — Assumptions of Linear Regression / Model evaluation*)

The residuals-versus-fitted plot (`06_residuals.png`) shows clear **heteroscedasticity**: spread
widens with predicted price. For a linear model this would violate the constant-variance assumption
and invalidate inference on the coefficients; for the tree ensembles used here it does not affect
validity, but it does explain why RMSE (≈ 80) is three times MAE (≈ 26) — a small number of large
errors on expensive properties dominates the squared metric. This asymmetry is the honest
characterisation of the model: **typical error is ~26 Lakhs; the RMSE headline is set by the tail.**

---

## 8. Submission

`output/submission/PES1PGE25DS037_submission.csv` — 2,664 rows, columns `ID,price`.

The final model is refitted on all 10,375 structurally clean training rows before predicting the
test set. The notebook asserts, against `sample_submission.csv` read at runtime, that column names,
column order, dtypes (`int64`, `float64`), row count and the `ID` sequence match exactly, and that
no prediction is missing or non-positive.

> **Specification discrepancy.** `docs/problem_statement.md` states the columns are `ID` / `Price`,
> but `sample_submission.csv` actually has `ID,price` with a lower-case `price`. The file is the
> authoritative contract, so the header is read from it at runtime rather than hard-coded.

Predicted test distribution: mean 109.77, median 73.55, min 12.17, max 2,018.97 Lakhs — closely
tracking the training distribution (mean 112.21, median 72.00), with no evidence of the compression
towards the mean that would indicate under-fitting.

---

## 9. Limitations and further work

1. **Rent coverage.** `avg_rent.csv` lists only 157 localities; even with fuzzy matching ~41 % of
   rows have no rent value and rely on median imputation plus a missingness flag.
2. **Fuzzy matching is heuristic.** A wrong substring match silently assigns a neighbouring
   locality's rent or distance. A curated locality gazetteer with geocoordinates would eliminate
   this risk and enable true spatial features (k-nearest-locality price levels).
3. **Range midpoints discard information.** `"2100 - 2850"` becomes 2,475, losing the uncertainty
   the range expresses; a range-width feature was not explored.
4. **No temporal dimension.** Listing dates are absent, so market drift cannot be modelled;
   `availability` is only a weak proxy for it.
5. **The tail dominates the metric.** A two-stage model (classify "premium vs standard", then apply
   a segment-specific regressor) is the natural next step, connecting directly to the classification
   techniques of *SMLC Units 1–4*.
6. **Hyperparameters were only lightly tuned.** A `RandomizedSearchCV` over the Gradient Boosting
   grid (*SMLR Session 5 — Hyperparameter Tuning: Grid Search, Random Search*) scored 84.23 against
   83.32 for the hand-set configuration, so the simpler settings were kept; a wider search and
   Bayesian optimisation were not attempted.
7. **`society` is encrypted**, so no external enrichment (builder reputation, amenities, age) is
   possible — only frequency and a target-encoded level effect are usable.

---

## 10. Course material applied

| Course reference | Where it was applied |
|---|---|
| SMLR S1 — Missing Values (standard & non-standard) | §2.1–2.2: informative-missingness flags; `total_sqft` non-standard values recovered instead of coerced to `NaN` |
| SMLR S1 — One-Hot / Label / Ordinal Encoding | §3.4: one-hot for `area_type`/`size_kind`; ordinal explicitly rejected for `location` |
| SMLR S1 — Outliers Detection/removal (Boxplot, IQR, Z-score) | §4: IQR rule implemented, measured, and rejected with evidence |
| SMLR S1 — Normalization and Transformation | §6: `log1p` target transform + Duan smearing |
| SMLR S1 — Train Test Split | §5: 80/20 split stratified on price deciles |
| SMLR S2 — Covariance & Correlation | §2.4: Pearson vs Spearman diagnostic on `sqft` |
| SMLR S3 — Model evaluation metrics; assumption tests | §7: RMSE/MAE/R²; residual heteroscedasticity diagnosis |
| SMLR S3 — Interaction effect | §3.3: `rent_x_sqft`, `dist_x_sqft` |
| SMLR S4 — Feature Extraction / Transformation / Scaling | §3: all parsers, derived ratios, `StandardScaler` for linear models |
| SMLR S5 — Regularization (Ridge, Lasso) | §6: `RidgeCV`/`LassoCV` benchmarked; bias-limited conclusion |
| SMLR S5 — Bias and Variance; Model Validation | §6–7: bias-limited linear models; 5-fold CV; variance reduction via blending |
| SMLR S5 — Hyperparameter Tuning (Grid/Random Search) | §6: `RandomizedSearchCV` over the boosting grid; result documented and rejected on evidence |
| SMLR S6 — Create ML pipeline; Packaging for reproducibility | §3.4: `Pipeline` + `ColumnTransformer`; pinned `requirements.txt` |
| SMLC U4 — Ensemble Learning: Random Forest, Bagging, AdaBoost, Gradient Boosting | §6: full ensemble benchmark; bagging-vs-boosting error decorrelation motivates the blend |
| SMLC U4 — Feature Importance | §7: impurity-based importance ranking |

---

## Figures

| File | Content |
|---|---|
| `figures/01_target_distribution.png` | Raw vs `log1p` price distribution |
| `figures/02_location_areatype.png` | Top-15 locations; price by `area_type` |
| `figures/03_external_features.png` | Price vs sqft, vs distance, vs locality rent |
| `figures/04_ppsqft_boxplot.png` | Price-per-sqft boxplot (IQR view) |
| `figures/05_model_comparison.png` | Validation RMSE by model and target scale |
| `figures/06_residuals.png` | Residuals vs fitted; predicted vs actual |
| `figures/07_feature_importance.png` | Top-15 Random Forest feature importances |

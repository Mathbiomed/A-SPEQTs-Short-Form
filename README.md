# A-SPEQTs Short Form

This repository contains the dataset and Python code used to select and validate a 14-item short form of the A-SPEQTs questionnaire. The short form was developed to predict the five full-scale A-SPEQTs domain scores using a smaller set of questionnaire items.

## Repository structure

```text
.
├── README.md
├── data/
│   └── A_SPEQT.xlsx
├── item_selection.py
└── validation.py
```

## Dataset

### `data/A_SPEQT.xlsx`

The dataset contains questionnaire responses from 500 adults with insomnia who were taking sleep medication.

- Each row represents one participant.
- The workbook contains one worksheet, `Sheet1`.
- The dataset contains 500 rows and 154 variables.
- The modeling analyses use 47 A-SPEQTs questionnaire items as candidate predictors and five full-scale mean scores as target variables.
- All questionnaire items used in the analysis are coded from 0 to 10.

### Candidate questionnaire items and target scores

| Domain | Candidate items | Number of items | Target variable | Target-score calculation |
|---|---|---:|---|---|
| SPRIS | `SPRIS_1`–`SPRIS_6` | 6 | `SPRIS_av` | Mean of `SPRIS_1`–`SPRIS_6` |
| SPAIS | `SPAIS_1`–`SPAIS_6` | 6 | `SPAIS_av` | Mean of `SPAIS_1`–`SPAIS_6` |
| SMYTH | `SMYTH_1`–`SMYTH_10` | 10 | `SMYTH_av` | Mean of `SMYTH_1`–`SMYTH_10` |
| SMORE | `SMORE_1`–`SMORE_10` | 10 | `SMORE_av` | Mean of `SMORE_1`–`SMORE_10` |
| SATIS | `SATIS_1`–`SATIS_12`, `SATIS_13_Global_item`, `SATIS_14`, and `SATIS_15` | 15 | `SATIS_av` | Mean of the 14 non-global items: `SATIS_1`–`SATIS_12`, `SATIS_14`, and `SATIS_15` |

`SATIS_13_Global_item` is included as a candidate predictor during item selection but is not included when calculating `SATIS_av`.

### Data preprocessing

The scripts select the variables needed for each analysis and apply complete-case filtering using `dropna()`.

- `item_selection.py` requires complete values for all 47 candidate items and the five target scores.
- `validation.py` requires complete values for the final 14 selected items and the five target scores.
- All 500 participants meet both complete-case definitions in the supplied dataset.
- No observations are removed from the current analyses.
- No missing-value imputation, standardization, normalization, or other predictor transformation is performed.

### Final 14 selected items

The final items, in selection order, are:

1. `SMORE_6`
2. `SPRIS_6`
3. `SMYTH_8`
4. `SPAIS_5`
5. `SATIS_15`
6. `SPRIS_3`
7. `SMYTH_4`
8. `SATIS_5`
9. `SMORE_4`
10. `SPAIS_3`
11. `SPRIS_5`
12. `SATIS_9`
13. `SMYTH_3`
14. `SMORE_8`

## Code

### `item_selection.py`

This script performs greedy questionnaire-item selection using CatBoost regression and five-fold out-of-fold (OOF) predictions.

The selection procedure is:

1. Evaluate every candidate item individually.
2. Select the first item with the largest R² values across the five target scales.
3. Identify the target scale with the lowest R².
4. From the remaining items, select the item that produces the largest improvement in R² for that weakest target.
5. Repeat Steps 3–4 until the minimum cross-validated R² across the five target scales reaches or exceeds the predefined threshold of 0.85. (This criterion was first satisfied after 14 items had been selected, defining the final short form)
* The script continues the selection procedure to 16 items for diagnostic examination of the performance trajectory. Items 15 and 16 are not included in the final short form evaluated in `validation.py`.

The same shuffled five-fold partitions are reused for all candidate item sets, with `random_state=42`. A separate CatBoost model is fitted for each of the five target scales.

The script saves intermediate selection states in:

```text
cache_runs/state_k1.pkl
cache_runs/state_k16.pkl
```

These files contain the selected item sequence, performance history, and cached feature-set evaluations.

### `validation.py`

This script evaluates the final 14-item short form.

The main validation analysis:

- divides the 500 participants into a 70% training set and a 30% test set;
- uses `random_state=42`, producing 350 training observations and 150 test observations;
- fits a separate CatBoost regression model for each target scale; and
- reports training R², test R², test mean squared error, and test root mean squared error.

The script saves the observed-versus-predicted plots as:

```text
scatterplot_final.png
scatter_plots11.svg
```

The script also performs a robustness analysis using 1,000 repeated 70/30 train-test splits. Split seeds range from 0 to 999. Within each training partition, five-fold cross-validation is performed, and predictions from the five fold-specific models are averaged for the corresponding test set.

The repeated-split results are saved as:

```text
robustness_1cv_results.csv
```

This output contains one row for each repetition and target scale, for a total of 5,000 rows. Its columns are:

| Column | Description |
|---|---|
| `iteration` | Repeated-split index and random seed, from 0 to 999 |
| `target` | Target A-SPEQTs scale |
| `cv_train_r2` | Mean validation-fold R² across the five folds |
| `cv_test_r2` | R² for the averaged model predictions on the held-out test set |
| `mse` | Test-set mean squared error |
| `rmse` | Test-set root mean squared error |

## CatBoost model parameters

The following parameters are used in the analyses:

| Parameter | Value |
|---|---:|
| `iterations` | 8000 |
| `learning_rate` | 0.002887502882062776 |
| `depth` | 3 |
| `l2_leaf_reg` | 9.576593911651704 |
| `bagging_temperature` | 0.03296057975269804 |
| `random_strength` | 2.1363388482181613 |
| `rsm` | 0.8897141011558393 |
| `min_data_in_leaf` | 28 |
| `bootstrap_type` | `Bayesian` |
| loss function | `RMSE` |
| model seed | 42 |

`item_selection.py` additionally uses early stopping with 200 rounds. The final models in `validation.py` use the fixed 8,000-iteration specification without early stopping.

## Requirements

The code requires Python and the following packages:

```text
numpy
pandas
openpyxl
joblib
scikit-learn
catboost
matplotlib
```

The packages can be installed using:

```bash
pip install numpy pandas openpyxl joblib scikit-learn catboost matplotlib
```

## How to run

Place `A_SPEQT.xlsx` inside the `data` directory and run the scripts from the repository root:

```bash
python item_selection.py
python validation.py
```

Both analyses can require substantial computation because separate CatBoost models are fitted repeatedly for five target scales.


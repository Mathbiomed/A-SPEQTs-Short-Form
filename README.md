# A-SPEQTs Short Form

This repository contains the dataset and Jupyter notebooks used to select and validate a 14-item short form of the A-SPEQTs questionnaire. The short form was developed to predict five full-scale A-SPEQTs domain scores from a reduced set of questionnaire items.

## Repository structure

```text
.
├── README.md
├── A_SPEQT.xlsx
├── item_selection.ipynb
└── validation.ipynb
```

The notebooks create additional cache, figure, and result files when run.

## Dataset

### `A_SPEQT.xlsx`

The workbook contains questionnaire responses from 500 adults with insomnia who were taking sleep medication.

- Each row represents one participant.
- The dataset contains 500 rows and 154 variables.
- The modeling analyses use 47 A-SPEQTs questionnaire items as candidate predictors and five full-scale mean scores as target variables.
- All 47 candidate items are integer coded from 0 to 10.
- The 47 candidate items and five target scores contain no missing values in the supplied dataset. Other variables in the workbook may contain missing values but are not used by these notebooks.

### Candidate questionnaire items and target scores

| Domain | Candidate items | Number of items | Target variable | Target-score calculation |
|---|---|---:|---|---|
| SPRIS | `SPRIS_1`–`SPRIS_6` | 6 | `SPRIS_av` | Mean of `SPRIS_1`–`SPRIS_6` |
| SPAIS | `SPAIS_1`–`SPAIS_6` | 6 | `SPAIS_av` | Mean of `SPAIS_1`–`SPAIS_6` |
| SMYTH | `SMYTH_1`–`SMYTH_10` | 10 | `SMYTH_av` | Mean of `SMYTH_1`–`SMYTH_10` |
| SMORE | `SMORE_1`–`SMORE_10` | 10 | `SMORE_av` | Mean of `SMORE_1`–`SMORE_10` |
| SATIS | `SATIS_1`–`SATIS_12`, `SATIS_13_Global_item`, `SATIS_14`, and `SATIS_15` | 15 | `SATIS_av` | Mean of the 14 non-global items: `SATIS_1`–`SATIS_12`, `SATIS_14`, and `SATIS_15` |

`SATIS_13_Global_item` is included as a candidate predictor during item selection but is not included when calculating `SATIS_av`.

## Final 14 selected items

The items evaluated as the final short form, listed in selection order, are:

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

The final short form contains three SPRIS items, two SPAIS items, three SMYTH items, three SMORE items, and three SATIS items.

## Analysis notebooks

### `item_selection.ipynb`

This notebook performs greedy questionnaire-item selection using CatBoost regression and five-fold out-of-fold (OOF) predictions.

The selection procedure is:

1. Evaluate each of the 47 candidate items individually.
2. Select the first item that maximizes the sum of the target-specific R² values.
3. Using the currently selected items, identify the target scale with the lowest R².
4. Evaluate each remaining candidate after adding it to the selected set.
5. Select the candidate that produces the largest improvement in R² for the current weakest-performing target.
6. Repeat Steps 3–5 to obtain the complete selection trajectory.

 A separate CatBoost model is fitted for each of the five target scales. The notebook memoizes previously evaluated feature sets to avoid unnecessary refitting.

The supplied implementation runs the selection sequence in three stages: `k=1`, `k=2–9`, and `k=10–16`. The first 14 selected items define the short form evaluated in `validation.ipynb`; items 15 and 16 are retained only in the displayed selection trajectory. The notebook uses a fixed `K_total=16` and does not implement an automatic R²-based stopping rule.

Intermediate selection states are saved as:

```text
cache_runs/state_k1.pkl
cache_runs/state_k9.pkl
```

The final plotting cell saves:

```text
figures/oof_r2_by_k.png
figures/oof_r2_by_k.pdf
figures/oof_r2_by_k_mean.png
figures/oof_r2_by_k_mean.pdf
```

### `validation.ipynb`

This notebook evaluates the final 14-item short form using five separate CatBoost regression models, one for each target scale.

#### Primary validation analysis

- The 500 participants are divided into a 70% training set and a 30% test set, producing 350 training observations and 150 test observations.
- Five-fold cross-validation is conducted within the training set to calculate the mean validation-fold R² for each target.
- A final model for each target is fitted using all 350 training observations.
- Final performance is evaluated on the 150 held-out test observations using R², mean squared error (MSE), and root mean squared error (RMSE).
- Observed-versus-predicted plots are saved as `scatterplot_final.png`.

#### Repeated-split robustness analysis

- The notebook repeats a 70%/30% train-test split 1,000 times, using split seeds from 0 to 999.
- For each split, one CatBoost model is fitted for each of the five target scales using the training partition and evaluated on the corresponding test partition.
- The analysis therefore produces 5,000 target-by-repetition result rows.
- R², MSE, and RMSE are saved to `robustness_results.csv`.
- The R² and MSE distributions are visualized in `boxplot_robustness.png`.

The columns in `robustness_results.csv` are:

| Column | Description |
|---|---|
| `iteration` | Repetition number from 1 to 1,000; the corresponding split uses `random_state=iteration - 1` |
| `target` | Target A-SPEQTs scale |
| `r2` | Test-set R² |
| `mse` | Test-set mean squared error |
| `rmse` | Test-set root mean squared error |

## Relationship to the manuscript figures

| Manuscript figure | Source notebook | Notebook output | Description |
|---|---|---|---|
| Figure 3(b) | `item_selection.ipynb` | `figures/oof_r2_by_k.png` / `.pdf` | First figure from the final plotting cell: target-specific OOF R² across the item-selection sequence |
| Figure 3(c) | `validation.ipynb` | `scatterplot_final.png` | First validation figure: observed versus predicted scores for the five target scales |
| Figure 4(d) | `validation.ipynb` | `boxplot_robustness.png` | Second validation figure: distributions of R² and MSE across 1,000 repeated train-test splits |

`figures/oof_r2_by_k_mean.png` and `.pdf` are diagnostic mean-OOF-R² plots and are not among the manuscript figures listed above.

## CatBoost model parameters

The following parameters are used in both notebooks:

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
| `bootstrap_type` | Explicitly set to `Bayesian` by the model-construction function in `item_selection.ipynb`; not explicitly set in `validation.ipynb` |
| loss function | `RMSE` |
| model seed | 42 |
| thread count | `-1` |

`item_selection.ipynb` supplies a validation fold to CatBoost and uses early stopping with 200 rounds. Although `early_stopping_rounds=200` is included in the parameter dictionary in `validation.ipynb`, those model fits do not supply an evaluation set; consequently, the validation and robustness models use the fixed 8,000-iteration specification.

## Reproducibility note

Item selection is performed using all 500 participants before `validation.ipynb` creates its 70%/30% split. The 150-person test partition is held out from final CatBoost model fitting, but it is not independent of the preceding item-selection stage. This distinction should be considered when interpreting the fixed-split test results.

## Requirements

The notebooks require Python, Jupyter, and the following packages:

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
pip install jupyter numpy pandas openpyxl joblib scikit-learn catboost matplotlib
```

## How to run

Place `A_SPEQT.xlsx`, `item_selection.ipynb`, and `validation.ipynb` in the same directory. The notebooks expect the workbook to be named exactly `A_SPEQT.xlsx`.

Run the notebooks in this order:

1. Open `item_selection.ipynb` and run all cells from top to bottom.
2. Open `validation.ipynb` and run all cells from top to bottom.

The item-selection and 1,000-repetition robustness analyses can require substantial computation because they repeatedly fit separate CatBoost models for five target scales.

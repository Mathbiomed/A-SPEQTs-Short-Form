import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from catboost import CatBoostRegressor



# Keep text editable when exporting figures as SVG
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'

# =========================================================
# Data and model setup
# =========================================================

df = pd.read_excel("data/A_SPEQT.xlsx")

# Target scale scores
target_cols = [
    "SPRIS_av",
    "SPAIS_av",
    "SMYTH_av",
    "SMORE_av",
    "SATIS_av"
]


# Final 14 questionnaire items selected by the greedy algorithm
selected_items = [
    'SMORE_6',   # 1
    'SPRIS_6',   # 2
    'SMYTH_8',   # 3
    'SPAIS_5',   # 4
    'SATIS_15',  # 5
    'SPRIS_3',   # 6
    'SMYTH_4',   # 7
    'SATIS_5',   # 8
    'SMORE_4',   # 9
    'SPAIS_3',   # 10
    'SPRIS_5',   # 11
    'SATIS_9',   # 12
    'SMYTH_3',   # 13
    'SMORE_8',   # 14
]


# CatBoost hyperparameters determined in the model selection stage
study_best_params = dict(
    iterations=8000,
    learning_rate=0.002887502882062776,
    depth=3,
    l2_leaf_reg=9.576593911651704,
    bagging_temperature=0.03296057975269804,
    random_strength=2.1363388482181613,
    rsm=0.8897141011558393,
    min_data_in_leaf=28
)


# =========================================================
# Prepare analysis data
# =========================================================

# Keep complete cases for the selected items and target scales
analysis_df = df[
    selected_items + target_cols
].dropna().copy()


# Predictor and target variables
X = analysis_df[selected_items]
y = analysis_df[target_cols]


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


print(f"Training sample: {len(X_train)}")
print(f"Test sample: {len(X_test)}")



# =========================================================
# Final model setup
# =========================================================

# Use the previously selected CatBoost hyperparameters
# and add the settings used for final model fitting.
best_params = dict(study_best_params)
best_params.update({
    "bootstrap_type":      "Bayesian",
    "loss_function":       "RMSE",
    "verbose":             False,
    "allow_writing_files": False,
    "random_seed":         42
})


# Color palette for target-specific scatter plots
# from dark navy to light blue
navy_to_blue = ['#1b2a6b', '#2e5fa3', '#4a90d9', '#82b8ef', '#c0d9f7']


# =========================================================
# Initialize figure
# =========================================================

n_targets = len(target_cols)
fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5))

# Convert axes to a list when only one target is used
if n_targets == 1:
    axes = [axes]


# =========================================================
# Final model evaluation
# =========================================================

print("\n" + "=" * 85)
print(
    f"{'Target Scale':<15} | "
    f"{'Train R2':<10} | "
    f"{'Test R2':<10} | "
    f"{'MSE':<10} | "
    f"{'RMSE':<10}"
)
print("-" * 85)

test_metrics = []


# Fit and evaluate one CatBoost model for each target scale
for i, t in enumerate(target_cols):

    # Fit the final model on the training set
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X_train, y_train[t], verbose=False)

    # Generate predictions for training and test sets
    tr_preds = final_model.predict(X_train)
    te_preds = final_model.predict(X_test)

    # Calculate model performance
    tr_r2   = r2_score(y_train[t], tr_preds)
    te_r2   = r2_score(y_test[t], te_preds)
    te_mse  = mean_squared_error(y_test[t], te_preds)
    te_rmse = np.sqrt(te_mse)

    test_metrics.append([te_r2, te_mse, te_rmse])

    print(
        f"{t:<15} | "
        f"{tr_r2:.4f}   | "
        f"{te_r2:.4f}   | "
        f"{te_mse:.4f}   | "
        f"{te_rmse:.4f}"
    )

    ax = axes[i]

    # Apply a target-specific color
    current_color = navy_to_blue[i] if i < len(navy_to_blue) else '#3498db'

    # Plot observed versus predicted scores
    ax.scatter(
        y_test[t],
        te_preds,
        alpha=0.7,
        color=current_color,
        edgecolors='white',
        s=60,
        zorder=2
    )

    # Use the same axis range across all target scales
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])

    # Target scale name
    ax.set_title(t, fontsize=12, fontweight='bold')

    # Reference line for perfect prediction (y = x)
    ax.plot(
        [0, 10],
        [0, 10],
        color='gray',
        linestyle='--',
        lw=1.5,
        zorder=1
    )

    # Transparent background and simplified plot appearance
    ax.set_facecolor('none')
    ax.grid(False)

    # Keep visible borders for later figure editing
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')


# Use a transparent background for the entire figure
fig.patch.set_facecolor('none')


# =========================================================
# Average test performance across target scales
# =========================================================

avg_metrics = np.mean(test_metrics, axis=0)

print("-" * 85)
print(
    f"{'Mean Score':<15} | "
    f"{'':10} | "
    f"{avg_metrics[0]:.4f}   | "
    f"{avg_metrics[1]:.4f}   | "
    f"{avg_metrics[2]:.4f}"
)
print("=" * 85)

plt.tight_layout()


# =========================================================
# Save final scatter plots
# =========================================================

# PNG for general use
plt.savefig(
    "scatterplot_final.png",
    dpi=300,
    transparent=True
)

# SVG for further editing in vector graphics software
plt.savefig(
    "scatter_plots11.svg",
    format="svg",
    transparent=True
)

plt.show()


# =========================================================
# Robustness analysis setup
# =========================================================

# Target scale scores
target_cols = [
    'SMORE_av',
    'SPRIS_av',
    'SMYTH_av',
    'SPAIS_av',
    'SATIS_av'
]

# Fixed 5-fold cross-validation scheme
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================================
# Combine the original training and test sets
# =========================================================

# Reconstruct the full dataset before repeatedly generating
# different random train/test partitions.
X_all = pd.concat(
    [X_train, X_test],
    axis=0
).reset_index(drop=True)

y_all = pd.concat(
    [y_train, y_test],
    axis=0
).reset_index(drop=True)



# =========================================================
# Repeated train/test split robustness check
# =========================================================

results = []

print("Robustness check started (1000 repetitions)...")


for i in range(1000):

    # Generate a different 70/30 train-test split
    # for each iteration.
    X_tr_all, X_te_all, y_tr_all, y_te_all = train_test_split(
        X_all[selected_items],
        y_all,
        test_size=0.3,
        random_state=i
    )


    # Use the same previously determined CatBoost
    # hyperparameters for every iteration.
    final_params = dict(study_best_params)
    final_params.update({
        "bootstrap_type":      "Bayesian",
        "loss_function":       "RMSE",
        "verbose":             False,
        "allow_writing_files": False,
        "random_seed":         42
    })


    # Fit and evaluate a separate model for each target scale
    for t in target_cols:

        fold_val_r2 = []
        te_preds = np.zeros(len(X_te_all))


        # =================================================
        # 5-fold CV within the training partition
        # =========================================================

        for train_idx, val_idx in kf.split(X_tr_all):

            X_tr  = X_tr_all.iloc[train_idx]
            X_val = X_tr_all.iloc[val_idx]

            y_tr  = y_tr_all[t].iloc[train_idx]
            y_val = y_tr_all[t].iloc[val_idx]


            # Fit a CatBoost model for the current fold
            fold_model = CatBoostRegressor(**final_params)
            fold_model.fit(
                X_tr,
                y_tr,
                verbose=False
            )


            # Evaluate the current fold on its validation set
            fold_val_r2.append(
                r2_score(
                    y_val,
                    fold_model.predict(X_val)
                )
            )


            # Accumulate test predictions from each fold model.
            # The final prediction is the average of five models.
            te_preds += fold_model.predict(X_te_all) / 5


        # Average validation R² across the five folds
        cv_tr_r2 = np.mean(fold_val_r2)

        # Evaluate the ensemble prediction on the test partition
        te_r2   = r2_score(y_te_all[t], te_preds)
        te_mse  = mean_squared_error(y_te_all[t], te_preds)
        te_rmse = np.sqrt(te_mse)


        # Store results from the current iteration and target
        results.append({
            'iteration':   i,
            'target':      t,
            'cv_train_r2': cv_tr_r2,
            'cv_test_r2':  te_r2,
            'mse':         te_mse,
            'rmse':        te_rmse
        })


    # Print progress every 20 iterations
    if (i + 1) % 20 == 0:
        print(f"Progress: {i+1}/1000 completed...")


# =========================================================
# Collect robustness results
# =========================================================

df_results = pd.DataFrame(results)


# =========================================================
# Summarize performance across repeated splits
# =========================================================

metrics_list = [
    'cv_train_r2',
    'cv_test_r2',
    'mse',
    'rmse'
]

print("\n" + "=" * 65)
print(
    f"Robustness summary by target scale "
    f"(1000 repeated splits) - {len(selected_items)} selected items"
)
print("=" * 65)


# Report the mean and standard deviation for each target scale
summary = (
    df_results
    .groupby('target')[metrics_list]
    .agg(['mean', 'std'])
    .round(4)
)

print(summary.to_string())
print("-" * 65)


# =========================================================
# Save robustness results
# =========================================================

df_results.to_csv(
    "robustness_1cv_results.csv",
    index=False
)

print("Robustness results saved to robustness_1cv_results.csv")
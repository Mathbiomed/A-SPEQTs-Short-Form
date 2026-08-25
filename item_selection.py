import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt


# =========================================================
# CatBoost model setup
# =========================================================
# Build a CatBoost model using the specified hyperparameters.
# Additional settings are applied when Bayesian bootstrap
# and early stopping are used.
def make_cat_model_from_user_params(user_params: dict):
    p = dict(user_params)

    if "bagging_temperature" in p:
        p["bootstrap_type"] = "Bayesian"

    if "early_stopping_rounds" in p:
        p.pop("od_wait", None)
        p.pop("od_type", None)

    return CatBoostRegressor(**p)


# =========================================================
# OOF evaluation
# =========================================================
# Evaluate a given set of questionnaire items using 5-fold
# out-of-fold (OOF) prediction for each target scale.
#
# Evaluation results are cached so that the same feature
# combination does not need to be fitted repeatedly.
class OOFCacheEvaluator:
    def __init__(self, base_df, target_cols, user_params,
                 n_splits=5, shuffle=True, random_state=42):

        self.base_df = base_df
        self.target_cols = list(target_cols)
        self.user_params = dict(user_params)

        # Use the same CV splits for all candidate feature sets
        idx = base_df.index.to_numpy()
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.splits = [(idx[tr], idx[va]) for tr, va in kf.split(idx)]

        # Cache: feature set -> {target: OOF_R2}
        self.cache = {}

    def eval_feature_set(self, features):
        # Sort feature names so that identical feature sets
        # share the same cache key regardless of order
        key = tuple(sorted(features))

        if key in self.cache:
            return self.cache[key]

        X = self.base_df[list(features)]
        out = {}

        # Fit a separate CatBoost model for each target scale
        for t in self.target_cols:
            y = self.base_df[t]
            oof_pred = pd.Series(index=self.base_df.index, dtype=float)

            # Generate OOF predictions using fixed 5-fold CV
            for tr_idx, va_idx in self.splits:
                model = make_cat_model_from_user_params(self.user_params)
                model.fit(
                    X.loc[tr_idx], y.loc[tr_idx],
                    eval_set=(X.loc[va_idx], y.loc[va_idx]),
                    use_best_model=True,
                    verbose=False
                )
                oof_pred.loc[va_idx] = model.predict(X.loc[va_idx])

            out[t] = float(r2_score(y, oof_pred))

        # Store results for reuse during greedy search
        self.cache[key] = out

        return out


# =========================================================
# Greedy selection - Step 1
# =========================================================
# Select the initial questionnaire item.
#
# Each candidate item is evaluated individually and the item
# with the largest sum of OOF R² across all target scales is selected.
def select_first_item_by_sum_oof(
    evaluator: OOFCacheEvaluator,
    candidate_items,
    target_cols
):
    candidate_items = list(candidate_items)
    target_cols = list(target_cols)

    best_item, best_sum, best_r2s = None, -np.inf, None

    # Evaluate every candidate as a single-item questionnaire
    for j in candidate_items:
        r2s = evaluator.eval_feature_set([j])
        score = sum(r2s[t] for t in target_cols)

        if score > best_sum:
            best_sum, best_item, best_r2s = score, j, r2s

    selected = [best_item]

    # Save the performance of the selected first item
    row = {
        "k": 1,
        "picked": best_item,
        "mode": "step1_sumOOF_R2",
        "sumOOF_R2": float(best_sum),
        "weakest_target": min(best_r2s, key=best_r2s.get),
        "delta_weakest": np.nan,
    }

    for t in target_cols:
        row[f"OOF_R2_{t}"] = float(best_r2s[t])

    hist = pd.DataFrame([row])

    return selected, hist


# =========================================================
# Greedy selection - Step 2 onward
# =========================================================
# After the first item is selected, identify the target scale
# with the lowest current OOF R² ("weakest target").
#
# Among the remaining questionnaire items, select the item
# that produces the largest improvement in the weakest target.
#
# The procedure is repeated until k_total items are selected.
# The function can resume from a previously saved selection state.
def continue_greedy_on_weakest_oof(
    evaluator: OOFCacheEvaluator,
    candidate_items,
    target_cols,
    k_total,
    start_selected,
    start_hist
):
    candidate_items = list(candidate_items)
    target_cols = list(target_cols)

    selected = list(start_selected)
    history = start_hist.copy()

    remaining = [x for x in candidate_items if x not in selected]

    while len(selected) < k_total and remaining:

        # Evaluate the currently selected questionnaire
        base_r2s = evaluator.eval_feature_set(selected)

        # Identify the target with the lowest OOF R²
        t_star = min(base_r2s, key=base_r2s.get)
        baseline = float(base_r2s[t_star])

        best_c, best_delta, best_new_r2s = None, -np.inf, None

        # Add each remaining item temporarily and evaluate
        # its improvement on the current weakest target
        for c in remaining:
            new_r2s = evaluator.eval_feature_set(selected + [c])
            delta = float(new_r2s[t_star] - baseline)

            if delta > best_delta:
                best_delta, best_c, best_new_r2s = delta, c, new_r2s

        # Permanently add the best candidate
        selected.append(best_c)
        remaining.remove(best_c)

        # Record performance after the new item is added
        row = {
            "k": len(selected),
            "picked": best_c,
            "mode": "stepk_delta_on_weakest_OOF",
            "weakest_target": t_star,
            "baseline_OOF_R2_weakest": baseline,
            "new_OOF_R2_weakest": float(best_new_r2s[t_star]),
            "delta_weakest": float(best_delta),
        }

        # Also record the total OOF R² across all target scales
        row["sumOOF_R2"] = float(sum(best_new_r2s[t] for t in target_cols))

        for t in target_cols:
            row[f"OOF_R2_{t}"] = float(best_new_r2s[t])

        history = pd.concat(
            [history, pd.DataFrame([row])],
            ignore_index=True
        )

        print(
            f"[k={len(selected):02d}] picked={best_c:>18s}  "
            f"weakest={t_star}  "
            f"baseline={baseline:.4f}  "
            f"new={best_new_r2s[t_star]:.4f}  "
            f"delta={best_delta:.4f}"
        )

    return selected, history


# =========================================================
# Plot OOF R² by number of selected items
# =========================================================
# Visualize how predictive performance changes as additional
# questionnaire items are selected.
#
# Two figures are generated:
#   1) Target-specific OOF R²
#   2) Mean OOF R² across all target scales
def plot_and_save_oof_curves(
    hist_df,
    target_cols,
    out_dir="figures",
    fname="oof_r2_by_k"
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = hist_df["k"].to_numpy()

    # Target-wise OOF R² curves
    plt.figure(figsize=(10, 5))

    for t in target_cols:
        plt.plot(
            k,
            hist_df[f"OOF_R2_{t}"],
            marker="o",
            linewidth=2,
            label=t
        )

    plt.xlabel("Number of selected items (k)")
    plt.ylabel("OOF R²")
    plt.xticks(k)
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_dir / f"{fname}.png", dpi=300)
    plt.savefig(out_dir / f"{fname}.pdf")
    plt.show()

    # Mean OOF R² across target scales
    plt.figure(figsize=(10, 5))

    vals = np.vstack(
        [hist_df[f"OOF_R2_{t}"] for t in target_cols]
    )

    plt.plot(
        k,
        vals.mean(axis=0),
        marker="o",
        linewidth=2
    )

    plt.xlabel("Number of selected items (k)")
    plt.ylabel("Mean OOF R²")
    plt.xticks(k)
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()

    plt.savefig(out_dir / f"{fname}_mean.png", dpi=300)
    plt.savefig(out_dir / f"{fname}_mean.pdf")
    plt.show()

    return str(out_dir)


# =========================================================
# Analysis setup
# =========================================================

# Candidate questionnaire items
candidate_items = [
    # SPRIS
    "SPRIS_1", "SPRIS_2", "SPRIS_3", "SPRIS_4", "SPRIS_5", "SPRIS_6",

    # SPAIS
    "SPAIS_1", "SPAIS_2", "SPAIS_3", "SPAIS_4", "SPAIS_5", "SPAIS_6",

    # SMYTH
    "SMYTH_1", "SMYTH_2", "SMYTH_3", "SMYTH_4", "SMYTH_5",
    "SMYTH_6", "SMYTH_7", "SMYTH_8", "SMYTH_9", "SMYTH_10",

    # SMORE
    "SMORE_1", "SMORE_2", "SMORE_3", "SMORE_4", "SMORE_5",
    "SMORE_6", "SMORE_7", "SMORE_8", "SMORE_9", "SMORE_10",

    # SATIS
    "SATIS_1", "SATIS_2", "SATIS_3", "SATIS_4", "SATIS_5", "SATIS_6",
    "SATIS_7", "SATIS_8", "SATIS_9", "SATIS_10", "SATIS_11", "SATIS_12",
    "SATIS_13_Global_item", "SATIS_14", "SATIS_15"
]


# Target scale scores to be reconstructed by the short-form items
target_cols = [
    "SPRIS_av",
    "SPAIS_av",
    "SMYTH_av",
    "SMORE_av",
    "SATIS_av"
]


# CatBoost hyperparameters
user_params = dict(
    loss_function="RMSE",
    iterations=8000,
    learning_rate=0.002887502882062776,
    depth=3,
    l2_leaf_reg=9.576593911651704,
    bagging_temperature=0.03296057975269804,
    random_strength=2.1363388482181613,
    rsm=0.8897141011558393,
    min_data_in_leaf=28,
    random_seed=42,
    early_stopping_rounds=200,
    verbose=200,
    thread_count=-1,
)


# =========================================================
# Fixed analysis sample
# =========================================================
# Use the same complete-case sample for every candidate
# feature combination so that OOF R² values are comparable.
fixed_cols = list(candidate_items) + list(target_cols)

df = pd.read_excel("data/A_SPEQT.xlsx")
base_df = df[fixed_cols].dropna().copy()

print("Fixed-base n =", len(base_df))


# =========================================================
# Initialize OOF evaluator
# =========================================================
# Five-fold CV splits are generated once and reused throughout
# the entire greedy selection procedure.
evaluator = OOFCacheEvaluator(
    base_df=base_df,
    target_cols=target_cols,
    user_params=user_params,
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================================
# PART 1
# Select the first item using the sum of OOF R²
# =========================================================
selected_1, hist_1 = select_first_item_by_sum_oof(
    evaluator=evaluator,
    candidate_items=candidate_items,
    target_cols=target_cols
)

print("\n[PART 1] Selected (k=1):", selected_1)


# Save the selected item, performance history,
# and cached model evaluations for later continuation.
os.makedirs("cache_runs", exist_ok=True)

joblib.dump(
    {
        "selected": selected_1,
        "hist": hist_1,
        "evaluator_cache": evaluator.cache
    },
    "cache_runs/state_k1.pkl"
)

print("Saved -> cache_runs/state_k1.pkl")


# =========================================================
# PART 2
# Continue greedy selection by improving the weakest target
# =========================================================

# End point of the second search stage
K_mid = 16


# Reload the state saved after selecting the first item
state = joblib.load("cache_runs/state_k1.pkl")

# Restore previously calculated OOF results
evaluator.cache.update(
    state.get("evaluator_cache", {})
)


# Continue selection from k=2 through k=K_mid
selected_mid, hist_mid = continue_greedy_on_weakest_oof(
    evaluator=evaluator,
    candidate_items=candidate_items,
    target_cols=target_cols,
    k_total=K_mid,
    start_selected=state["selected"],
    start_hist=state["hist"],
)

print(f"\n[PART 2] Selected (k=1..{K_mid}):")
print(selected_mid)


# Save the intermediate state so that the search
# can later be resumed without starting from k=1.
joblib.dump(
    {
        "selected": selected_mid,
        "hist": hist_mid,
        "evaluator_cache": evaluator.cache
    },
    f"cache_runs/state_k{K_mid}.pkl"
)

print(f"Saved -> cache_runs/state_k{K_mid}.pkl")
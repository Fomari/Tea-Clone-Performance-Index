"""
Modelling Pipeline 
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"
MODEL_DIR = "outputs/models"
for d in [FIG_DIR, TAB_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

SEED = 42
np.random.seed(SEED)

df = pd.read_csv("data/processed_quarterly_model_ready.csv")
df = df.sort_values(["clone_type", "year", "quarter"]).reset_index(drop=True)

FEATURES = [
    "price_lag1", "price_lag2", "price_rolling_mean3",
    "avg_bp1_share", "avg_rainfall_mm", "avg_temperature_c",
    "avg_exchange_rate", "is_q1", "is_q2", "is_q3", "is_q4",
]
TARGET = "avg_price_usd"

# One-hot encode clone identity so a single global model can learn
# clone-specific effects.
clone_dummies = pd.get_dummies(df["clone_type"], prefix="clone")
X_full = pd.concat([df[FEATURES], clone_dummies], axis=1)
y_full = df[TARGET]

# ---------------------------------------------------------------
# Time-based split: train on 2022Q3-2024Q4, test on 2025 quarters
# (per-clone, preserving chronology)
# ---------------------------------------------------------------
df["period_index"] = df["year"] * 4 + df["quarter"]
split_threshold = 2025 * 4 + 1  # test = 2025 Q1 onward

train_mask = df["period_index"] < split_threshold
test_mask = ~train_mask

X_train, X_test = X_full[train_mask], X_full[test_mask]
y_train, y_test = y_full[train_mask], y_full[test_mask]

print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
print("Test period(s):", sorted(df.loc[test_mask, "year_quarter"].unique()))


# ---------------------------------------------------------------
# Baseline 1: Naive persistence (predict price_lag1)
# ---------------------------------------------------------------
pred_naive = X_test["price_lag1"].values

# ---------------------------------------------------------------
# Baseline 2: Seasonal naive (same quarter, previous year average per clone)
# ---------------------------------------------------------------
seasonal_lookup = (
    df[df["year"] == 2024]
    .groupby(["clone_type", "quarter"])["avg_price_usd"].mean()
    .to_dict()
)
pred_seasonal = df.loc[test_mask].apply(
    lambda r: seasonal_lookup.get((r["clone_type"], r["quarter"]), r["price_lag1"]), axis=1
).values


# ---------------------------------------------------------------
# Model 3: Random Forest Regressor (with GridSearch tuning)
# ---------------------------------------------------------------
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [1, 2, 4],
}
tscv = TimeSeriesSplit(n_splits=4)
rf_search = GridSearchCV(
    RandomForestRegressor(random_state=SEED),
    rf_param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
pred_rf = rf_best.predict(X_test)

# ---------------------------------------------------------------
# Model 4: HistGradientBoostingRegressor (XGBoost-equivalent, tuned)
# ---------------------------------------------------------------
gbm_param_grid = {
    "max_iter": [100, 200, 300],
    "max_depth": [2, 3, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_regularization": [0.0, 0.1, 1.0],
}
gbm_search = GridSearchCV(
    HistGradientBoostingRegressor(random_state=SEED),
    gbm_param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
)
gbm_search.fit(X_train, y_train)
gbm_best = gbm_search.best_estimator_
pred_gbm = gbm_best.predict(X_test)


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


results = [
    evaluate("Naive Persistence (t-1)", y_test, pred_naive),
    evaluate("Seasonal Naive (prev. year same quarter)", y_test, pred_seasonal),
    evaluate("Random Forest Regressor (tuned)", y_test, pred_rf),
    evaluate("HistGradientBoosting (XGBoost-equivalent, tuned)", y_test, pred_gbm),
]
results_df = pd.DataFrame(results)
results_df.to_csv(f"{TAB_DIR}/model_comparison_results.csv", index=False)
print("\n=== Model Comparison ===")
print(results_df)

# Save best hyperparameters
hyperparams = {
    "RandomForest_best_params": rf_search.best_params_,
    "RandomForest_best_cv_MAE": -rf_search.best_score_,
    "HistGBM_best_params": gbm_search.best_params_,
    "HistGBM_best_cv_MAE": -gbm_search.best_score_,
}
with open(f"{TAB_DIR}/best_hyperparameters.json", "w") as f:
    json.dump(hyperparams, f, indent=2, default=str)
print("\nBest hyperparameters:\n", json.dumps(hyperparams, indent=2, default=str))


# ---------------------------------------------------------------
# Determine champion model
# ---------------------------------------------------------------
champion_row = results_df.loc[results_df["MAE"].idxmin()]
champion_name = champion_row["model"]
champion_preds = {
    "Naive Persistence (t-1)": pred_naive,
    "Seasonal Naive (prev. year same quarter)": pred_seasonal,
    "Random Forest Regressor (tuned)": pred_rf,
    "HistGradientBoosting (XGBoost-equivalent, tuned)": pred_gbm,
}[champion_name]
print(f"\nChampion model: {champion_name}")


# ---------------------------------------------------------------
# Figure 4.8: Predicted vs Actual (champion model)
# ---------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, champion_preds, alpha=0.7, edgecolor="k")
lims = [min(y_test.min(), champion_preds.min()), max(y_test.max(), champion_preds.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Avg. Price (USD/kg)")
plt.ylabel("Predicted Avg. Price (USD/kg)")
plt.title(f"Figure 4.8: Predicted vs Actual Quarterly Price\n({champion_name})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_8_predicted_vs_actual.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Figure 4.9: Residual distribution
# ---------------------------------------------------------------
residuals = y_test.values - champion_preds
plt.figure(figsize=(7.5, 5.4))
sns.histplot(residuals, kde=True, color="darkorange")
plt.axvline(0, color="k", linestyle="--")
plt.title("Figure 4.9: Residual Distribution\n(Champion Model: HistGradientBoosting)", fontsize=13)
plt.xlabel("Residual (Actual - Predicted) USD/kg")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_9_residual_distribution.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Figure 4.10: Feature importance (Random Forest)
# ---------------------------------------------------------------
importances = pd.Series(rf_best.feature_importances_, index=X_full.columns).sort_values(ascending=False)
importances.to_csv(f"{TAB_DIR}/rf_feature_importances.csv")
plt.figure(figsize=(8, 6))
importances.head(10).plot(kind="barh", color="teal")
plt.gca().invert_yaxis()
plt.title("Figure 4.10: Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_10_rf_feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Figure 4.11: Learning curve (training set size vs MAE) for champion-class model
# ---------------------------------------------------------------
from sklearn.model_selection import learning_curve
lc_model = rf_best if "Random Forest" in champion_name else gbm_best
train_sizes, train_scores, val_scores = learning_curve(
    lc_model, X_train, y_train, cv=tscv,
    scoring="neg_mean_absolute_error",
    train_sizes=np.linspace(0.3, 1.0, 5)
)
plt.figure(figsize=(7, 5))
plt.plot(train_sizes, -train_scores.mean(axis=1), marker="o", label="Training MAE")
plt.plot(train_sizes, -val_scores.mean(axis=1), marker="o", label="Validation MAE")
plt.xlabel("Training Set Size")
plt.ylabel("MAE (USD/kg)")
plt.title(f"Figure 4.11: Learning Curve ({champion_name.split('(')[0].strip()})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_11_learning_curve.png", dpi=150)
plt.close()


# ---------------------------------------------------------------
# Predictions table for 2025 test quarters (per clone)
# ---------------------------------------------------------------
pred_table = df.loc[test_mask, ["clone_type", "year_quarter", "avg_price_usd"]].copy()
pred_table["predicted_price_naive"] = pred_naive
pred_table["predicted_price_seasonal"] = pred_seasonal
pred_table["predicted_price_rf"] = pred_rf
pred_table["predicted_price_gbm"] = pred_gbm
pred_table["champion_prediction"] = champion_preds
pred_table.to_csv(f"{TAB_DIR}/quarterly_predictions_2025.csv", index=False)
print("\nPredictions table:\n", pred_table)


# Save champion model
import joblib
joblib.dump(rf_best, f"{MODEL_DIR}/random_forest_model.joblib")
joblib.dump(gbm_best, f"{MODEL_DIR}/hist_gbm_model.joblib")
with open(f"{MODEL_DIR}/champion_model.txt", "w") as f:
    f.write(champion_name)

print("\nModelling complete. Outputs saved to outputs/")
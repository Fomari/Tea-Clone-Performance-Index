"""
Prescriptive Analytics Pipeline 
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"

# ---------------------------------------------------------------
# Assumptions (documented for transparency / reproducibility)
# ---------------------------------------------------------------
AVG_YIELD_KG_PER_HA_PER_QUARTER = 1500   # green leaf -> made tea kg, smallholder average
PRODUCTION_COST_USD_PER_KG = 1.35        # average cost of production per kg made tea
KTDA_BONUS_SHARE = 0.45                  # share of net surplus distributed as bonus

quarterly = pd.read_csv("data/processed_quarterly_clone_panel.csv")
predictions = pd.read_csv(f"{TAB_DIR}/quarterly_predictions_2025.csv")

# ---------------------------------------------------------------
# 1. Historical stability metrics per clone (full history)
# ---------------------------------------------------------------
stability = quarterly.groupby("clone_type").agg(
    historical_avg_price=("avg_price_usd", "mean"),
    historical_price_std=("avg_price_usd", "std"),
).reset_index()
stability["coefficient_of_variation"] = (
    stability["historical_price_std"] / stability["historical_avg_price"]
)

# ---------------------------------------------------------------
# 2. Forecasted 2025 economics per clone (using champion predictions)
# ---------------------------------------------------------------
forecast = predictions.groupby("clone_type").agg(
    forecast_avg_price_2025=("champion_prediction", "mean")
).reset_index()

econ = forecast.merge(stability, on="clone_type")

# Annual revenue per hectare = forecast price * yield * 4 quarters
econ["annual_revenue_usd_per_ha"] = econ["forecast_avg_price_2025"] * AVG_YIELD_KG_PER_HA_PER_QUARTER * 4
econ["annual_cost_usd_per_ha"] = PRODUCTION_COST_USD_PER_KG * AVG_YIELD_KG_PER_HA_PER_QUARTER * 4
econ["annual_net_surplus_usd_per_ha"] = econ["annual_revenue_usd_per_ha"] - econ["annual_cost_usd_per_ha"]
econ["expected_annual_bonus_usd_per_ha"] = econ["annual_net_surplus_usd_per_ha"] * KTDA_BONUS_SHARE
econ["roi_pct"] = (econ["annual_net_surplus_usd_per_ha"] / econ["annual_cost_usd_per_ha"]) * 100

# ---------------------------------------------------------------
# 3. Composite ranking score
#    Normalised expected return (60%) minus normalised volatility (40%)
# ---------------------------------------------------------------
def normalise(s):
    return (s - s.min()) / (s.max() - s.min())

econ["norm_return"] = normalise(econ["annual_net_surplus_usd_per_ha"])
econ["norm_volatility"] = normalise(econ["coefficient_of_variation"])
econ["composite_score"] = 0.6 * econ["norm_return"] + 0.4 * (1 - econ["norm_volatility"])

econ = econ.sort_values("composite_score", ascending=False).reset_index(drop=True)
econ["rank"] = range(1, len(econ) + 1)

econ.to_csv(f"{TAB_DIR}/clone_economic_ranking.csv", index=False)
print("=== Clone Economic Ranking (2025 forecast) ===")
print(econ[["rank", "clone_type", "forecast_avg_price_2025",
             "coefficient_of_variation", "expected_annual_bonus_usd_per_ha",
             "roi_pct", "composite_score"]])

# ---------------------------------------------------------------
# Figure: Clone ranking bar chart (composite score)
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=econ, x="clone_type", y="composite_score",
            order=econ["clone_type"], color="darkgreen")
plt.title("Figure 4.12: Clone Performance Index - Composite Ranking Score (2025 Forecast)")
plt.ylabel("Composite Score (0-1, higher = better)")
plt.xlabel("Clone Type")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_12_composite_ranking.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Figure: Expected annual bonus per hectare by clone
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=econ, x="clone_type", y="expected_annual_bonus_usd_per_ha",
            order=econ["clone_type"], color="goldenrod")
plt.title("Figure 4.13: Expected Annual Bonus per Hectare by Clone (USD, 2025 Forecast)")
plt.ylabel("Expected Annual Bonus (USD/ha)")
plt.xlabel("Clone Type")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_13_expected_bonus.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Figure: Risk-Return scatter (volatility vs forecast price)
# ---------------------------------------------------------------
plt.figure(figsize=(7.5, 5.5))
for _, row in econ.iterrows():
    plt.scatter(row["coefficient_of_variation"], row["forecast_avg_price_2025"], s=100)
    plt.annotate(row["clone_type"], (row["coefficient_of_variation"], row["forecast_avg_price_2025"]),
                 xytext=(5, 5), textcoords="offset points")
plt.xlabel("Price Volatility (Coefficient of Variation)")
plt.ylabel("Forecast Avg. Price 2025 (USD/kg)")
plt.title("Figure 4.14: Risk-Return Profile of Tea Clones")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_14_risk_return_scatter.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Scenario analysis: what-if exchange rate shock (+/- 5%)
# ---------------------------------------------------------------
scenarios = []
for shock_name, shock_factor in [("Baseline", 1.0), ("KES Depreciation +5%", 1.05), ("KES Appreciation -5%", 0.95)]:
    temp = econ.copy()
    temp["scenario"] = shock_name
    temp["adj_annual_bonus_usd_per_ha"] = temp["expected_annual_bonus_usd_per_ha"] * shock_factor
    scenarios.append(temp[["clone_type", "scenario", "adj_annual_bonus_usd_per_ha"]])
scenario_df = pd.concat(scenarios)
scenario_df.to_csv(f"{TAB_DIR}/scenario_analysis_exchange_rate.csv", index=False)

plt.figure(figsize=(9, 5.5))
sns.barplot(data=scenario_df, x="clone_type", y="adj_annual_bonus_usd_per_ha", hue="scenario")
plt.title("Figure 4.15: Scenario Analysis - Exchange Rate Sensitivity of Expected Bonus")
plt.ylabel("Expected Annual Bonus (USD/ha)")
plt.xlabel("Clone Type")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_15_scenario_analysis.png", dpi=150)
plt.close()

print("\nPrescriptive analytics complete. Outputs saved to outputs/")
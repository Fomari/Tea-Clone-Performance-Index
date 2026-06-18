"""
Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="deep")
FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

lot = pd.read_csv("data/processed_lot_level.csv", parse_dates=["sale_date"])
qtr = pd.read_csv("data/processed_quarterly_clone_panel.csv")

# ---------------------------------------------------------------
# 1. Descriptive statistics tables
# ---------------------------------------------------------------
desc = lot[["auction_price_usd", "rainfall_mm", "temperature_c",
            "made_tea_kg", "bp1_share", "exchange_rate"]].describe().T
desc.to_csv(f"{TAB_DIR}/lot_level_descriptive_stats.csv")

clone_summary = lot.groupby("clone_type")["auction_price_usd"].agg(
    ["mean", "std", "min", "max", "count"]
).reset_index()
clone_summary.to_csv(f"{TAB_DIR}/clone_price_summary.csv", index=False)

grade_summary = lot.groupby("grade")["auction_price_usd"].agg(
    ["mean", "std", "count"]
).reset_index()
grade_summary.to_csv(f"{TAB_DIR}/grade_price_summary.csv", index=False)

# ---------------------------------------------------------------
# 2. Price distribution by clone (boxplot)
# ---------------------------------------------------------------
plt.figure(figsize=(9, 5.5))
order = lot.groupby("clone_type")["auction_price_usd"].median().sort_values(ascending=False).index
sns.boxplot(data=lot, x="clone_type", y="auction_price_usd", order=order)
plt.title("Figure 4.1: Auction Price Distribution by Tea Clone (2022-2025)")
plt.xlabel("Clone Type")
plt.ylabel("Auction Price (USD/kg)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_1_price_by_clone_boxplot.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Price distribution by grade
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
order_g = lot.groupby("grade")["auction_price_usd"].median().sort_values(ascending=False).index
sns.boxplot(data=lot, x="grade", y="auction_price_usd", order=order_g)
plt.title("Figure 4.2: Auction Price Distribution by Tea Grade")
plt.xlabel("Grade")
plt.ylabel("Auction Price (USD/kg)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_2_price_by_grade_boxplot.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Time series of quarterly average price per clone
# ---------------------------------------------------------------
plt.figure(figsize=(11, 6))
for clone in qtr["clone_type"].unique():
    sub = qtr[qtr["clone_type"] == clone].sort_values(["year", "quarter"])
    plt.plot(sub["year_quarter"], sub["avg_price_usd"], marker="o", label=clone)
plt.xticks(rotation=60)
plt.title("Figure 4.3: Quarterly Average Auction Price by Clone (2022-2025)")
plt.xlabel("Year-Quarter")
plt.ylabel("Average Auction Price (USD/kg)")
plt.legend(title="Clone")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_3_quarterly_price_trend_by_clone.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Correlation heatmap (lot-level numeric features)
# ---------------------------------------------------------------
num_cols = ["auction_price_usd", "exchange_rate", "rainfall_mm",
            "temperature_c", "made_tea_kg", "bp1_share"]
corr = lot[num_cols].corr()
corr.to_csv(f"{TAB_DIR}/correlation_matrix.csv")

plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Figure 4.4: Correlation Heatmap of Lot-Level Numeric Variables")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_4_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Seasonal price pattern (monthly average across years)
# ---------------------------------------------------------------
lot["month"] = pd.to_datetime(lot["sale_date"]).dt.month
monthly = lot.groupby("month")["auction_price_usd"].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(data=monthly, x="month", y="auction_price_usd", color="seagreen")
plt.title("Figure 4.5: Average Auction Price by Calendar Month (2022-2025)")
plt.xlabel("Month")
plt.ylabel("Average Auction Price (USD/kg)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_5_seasonal_price_pattern.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. BP1 share vs price scatter
# ---------------------------------------------------------------
plt.figure(figsize=(7.5, 5.5))
sns.scatterplot(data=lot.sample(min(3000, len(lot)), random_state=42),
                 x="bp1_share", y="auction_price_usd", hue="clone_type", alpha=0.5, s=18)
plt.title("Figure 4.6: BP1 Grade Share vs Auction Price")
plt.xlabel("BP1 Share")
plt.ylabel("Auction Price (USD/kg)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_6_bp1share_vs_price.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 8. Class / clone distribution (lot counts)
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=lot, x="clone_type",
               order=lot["clone_type"].value_counts().index, color="steelblue")
plt.title("Figure 4.7: Number of Auction Lots per Clone Type (2022-2025)")
plt.xlabel("Clone Type")
plt.ylabel("Number of Lots")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_4_7_lot_count_by_clone.png", dpi=150)
plt.close()

print("EDA complete. Figures saved to", FIG_DIR)
print("Tables saved to", TAB_DIR)
print("\nClone summary:\n", clone_summary)
print("\nCorrelation matrix:\n", corr)
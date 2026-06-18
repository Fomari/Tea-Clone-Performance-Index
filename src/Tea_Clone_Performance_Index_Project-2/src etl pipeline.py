"""
ETL Pipeline - The Tea Clone Performance Index
"""

import pandas as pd
import numpy as np
import os

RAW_AUCTION_PATH = "data/ktda_auction_files_2022-2025.csv"
CLONE_MAPPING_PATH = "data/clone_mapping.csv"
OUTPUT_DIR = "outputs/tables"
PROCESSED_LOT_PATH = "data/processed_lot_level.csv"
PROCESSED_QUARTERLY_PATH = "data/processed_quarterly_clone_panel.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_raw_data():
    """Load raw auction lot data and clone mapping reference table."""
    auctions = pd.read_csv(RAW_AUCTION_PATH)
    mapping = pd.read_csv(CLONE_MAPPING_PATH)
    return auctions, mapping
def clean_auction_data(df):
    """Standardise types, parse dates, remove duplicates, check missing values."""
    n_before = len(df)

    # Standardise column names 
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse dates - file uses M/D/YYYY format
    df["sale_date"] = pd.to_datetime(df["sale_date"], format="%m/%d/%Y", errors="coerce")

    # Drop rows with unparseable dates
    df = df.dropna(subset=["sale_date"])

    # Standardise categorical text fields
    for col in ["factory", "grade", "clone_type", "lot_number"]:
        df[col] = df[col].astype(str).str.strip()

    # Remove exact duplicate lot records
    n_dupes = df.duplicated(subset=["lot_number", "sale_date"]).sum()
    df = df.drop_duplicates(subset=["lot_number", "sale_date"])

    # Numeric sanity checks - clip impossible negative values
    numeric_cols = ["auction_price_usd", "exchange_rate", "rainfall_mm",
                     "temperature_c", "made_tea_kg", "bp1_share"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    df = df[(df["auction_price_usd"] > 0) & (df["made_tea_kg"] > 0)]

    # Outlier treatment on auction price using IQR capping (winsorisation)
    q1, q3 = df["auction_price_usd"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df["auction_price_usd"] < lower) | (df["auction_price_usd"] > upper)).sum()
    df["auction_price_usd"] = df["auction_price_usd"].clip(lower, upper)

    n_after = len(df)
    report = {
        "rows_before": n_before,
        "rows_after_dedup_and_clean": n_after,
        "duplicates_removed": int(n_dupes),
        "price_outliers_capped": int(n_outliers),
    }
    return df, report


def validate_clone_mapping(auctions, mapping):
    # Derive lot prefix 
    auctions = auctions.copy()
    auctions["lot_prefix"] = auctions["lot_number"].str.rsplit("-", n=1).str[0]

    merged = auctions.merge(
        mapping[["factory", "lot_prefix", "clone_type"]],
        on=["factory", "lot_prefix"],
        how="left",
        suffixes=("", "_ref"),
    )

    mismatches = merged[merged["clone_type"] != merged["clone_type_ref"]]
    unmatched = merged["clone_type_ref"].isna().sum()

    validation_report = {
        "total_lots": len(merged),
        "lots_matched_to_reference": int((~merged["clone_type_ref"].isna()).sum()),
        "lots_unmatched": int(unmatched),
        "clone_label_mismatches": int(len(mismatches)),
    }

    # Use the validated clone_type
    merged["clone_type_final"] = merged["clone_type_ref"].fillna(merged["clone_type"])
    merged["clone_type"] = merged["clone_type_final"]
    merged = merged.drop(columns=["clone_type_ref", "clone_type_final", "lot_prefix"])

    return merged, validation_report


def engineer_lot_features(df):
    """Add calendar and economic engineered features at lot level."""
    df = df.copy()
    df["year"] = df["sale_date"].dt.year
    df["quarter"] = df["sale_date"].dt.quarter
    df["month"] = df["sale_date"].dt.month
    df["year_quarter"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)

    # Seasonal indicator (Kenyan tea seasons: peak rain season ~ Mar-May & Oct-Dec)
    df["season"] = np.where(df["month"].isin([3, 4, 5, 10, 11, 12]), "high_rain", "dry")

    # Price converted to KES using the exchange rate of that auction date
    df["auction_price_kes"] = df["auction_price_usd"] * df["exchange_rate"]

    # Revenue proxy for the lot (price per kg * kg made)
    df["lot_revenue_usd"] = df["auction_price_usd"] * df["made_tea_kg"]

    return df


def build_quarterly_clone_panel(df):
    """
    Aggregate lot-level data into a quarterly clone-level panel - the
    primary modelling unit for the time-series forecasting task.
    """
    agg = df.groupby(["clone_type", "year", "quarter", "year_quarter"]).agg(
        avg_price_usd=("auction_price_usd", "mean"),
        median_price_usd=("auction_price_usd", "median"),
        price_std=("auction_price_usd", "std"),
        total_made_tea_kg=("made_tea_kg", "sum"),
        avg_bp1_share=("bp1_share", "mean"),
        avg_rainfall_mm=("rainfall_mm", "mean"),
        avg_temperature_c=("temperature_c", "mean"),
        avg_exchange_rate=("exchange_rate", "mean"),
        n_lots=("lot_number", "count"),
        total_revenue_usd=("lot_revenue_usd", "sum"),
    ).reset_index()

    agg["price_std"] = agg["price_std"].fillna(0)
    agg["coefficient_of_variation"] = agg["price_std"] / agg["avg_price_usd"]

    # Sort for time-series feature creation
    agg = agg.sort_values(["clone_type", "year", "quarter"]).reset_index(drop=True)

    # Lagged price features (t-1, t-2) and rolling mean, per clone
    agg["price_lag1"] = agg.groupby("clone_type")["avg_price_usd"].shift(1)
    agg["price_lag2"] = agg.groupby("clone_type")["avg_price_usd"].shift(2)
    agg["price_rolling_mean3"] = (
        agg.groupby("clone_type")["avg_price_usd"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    )
    agg["price_pct_change"] = agg.groupby("clone_type")["avg_price_usd"].pct_change()

    # Seasonal one-hot for quarter
    for q in [1, 2, 3, 4]:
        agg[f"is_q{q}"] = (agg["quarter"] == q).astype(int)

    # Drop rows without lag history (first 2 quarters per clone) for supervised modelling
    model_ready = agg.dropna(subset=["price_lag1", "price_lag2"]).reset_index(drop=True)

    return agg, model_ready


def run_pipeline():
    auctions, mapping = load_raw_data()
    auctions_clean, clean_report = clean_auction_data(auctions)
    auctions_validated, validation_report = validate_clone_mapping(auctions_clean, mapping)
    auctions_featured = engineer_lot_features(auctions_validated)

    quarterly_full, quarterly_model_ready = build_quarterly_clone_panel(auctions_featured)

    auctions_featured.to_csv(PROCESSED_LOT_PATH, index=False)
    quarterly_full.to_csv(PROCESSED_QUARTERLY_PATH, index=False)
    quarterly_model_ready.to_csv("data/processed_quarterly_model_ready.csv", index=False)

    print("=== Data Cleaning Report ===")
    for k, v in clean_report.items():
        print(f"{k}: {v}")
    print("\n=== Clone Mapping Validation Report ===")
    for k, v in validation_report.items():
        print(f"{k}: {v}")
    print(f"\nLot-level processed rows: {len(auctions_featured)}")
    print(f"Quarterly clone panel rows: {len(quarterly_full)}")
    print(f"Model-ready quarterly rows: {len(quarterly_model_ready)}")

    return auctions_featured, quarterly_full, quarterly_model_ready, clean_report, validation_report


if __name__ == "__main__":
    run_pipeline()
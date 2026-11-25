
import pandas as pd
import numpy as np
from pathlib import Path

# Load cleaned dataset
DATA_PATH = Path("datasets/processed/invoices_cleaned.parquet")
OUTPUT_DIR = Path("reports/week3_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

print("=" * 65)
print("WEEK 3 - COMPREHENSIVE FEATURE ENGINEERING AUDIT")
print("=" * 65)

# 1. Dataset overview
print("\n1. DATASET OVERVIEW")
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isna().sum().sum())

# 2. Identifier checks
print("\n2. IDENTIFIER CHECKS")
for col in ["invoice_id", "doc_id", "cust_number"]:
    if col in df.columns:
        print(f"{col}: unique={df[col].nunique()}, missing={df[col].isna().sum()}")

print("Duplicate invoice IDs:", df["invoice_id"].duplicated().sum())
print("Duplicate document IDs:", df["doc_id"].duplicated().sum())

# 3. Target distribution
print("\n3. TARGET DISTRIBUTION")
print(df["late_payment"].value_counts(dropna=False))
print("\nTarget percentages:")
print((df["late_payment"].value_counts(normalize=True, dropna=False) * 100).round(2))

# 4. Date feature audit
print("\n4. DATE FEATURE AUDIT")

date_cols = [
    "posting_date",
    "document_create_date",
    "document_create_date_alt",
    "due_in_date",
    "baseline_create_date",
    "clear_date"
]

for col in date_cols:
    print(
        col,
        "| missing:", df[col].isna().sum(),
        "| unique:", df[col].nunique(),
        "| min:", df[col].min(),
        "| max:", df[col].max()
    )

# 5. Candidate date feature distributions
print("\n5. CANDIDATE DATE FEATURES")

features = pd.DataFrame(index=df.index)

features["payment_term_days"] = (
    df["due_in_date"] - df["baseline_create_date"]
).dt.days

features["posting_to_due_days"] = (
    df["due_in_date"] - df["posting_date"]
).dt.days

features["baseline_to_posting_days"] = (
    df["posting_date"] - df["baseline_create_date"]
).dt.days

features["invoice_month"] = df["posting_date"].dt.month
features["invoice_quarter"] = df["posting_date"].dt.quarter
features["invoice_day_of_week"] = df["posting_date"].dt.dayofweek

print(features.describe().round(2).to_string())

print("\nFeature correlations:")
print(features.corr().round(3).to_string())

# 6. Categorical feature audit
print("\n6. CATEGORICAL FEATURE AUDIT")

categorical_cols = [
    "business_code",
    "invoice_currency",
    "document_type",
    "cust_payment_terms",
    "cust_number"
]

for col in categorical_cols:
    print(f"\n{col}")
    print("Unique:", df[col].nunique())
    print("Missing:", df[col].isna().sum())
    print("Top values:")
    print(df[col].value_counts(dropna=False).head(10).to_string())

# 7. Rare categories
print("\n7. RARE CATEGORY COUNTS")

for col in ["business_code", "document_type", "cust_payment_terms"]:
    counts = df[col].value_counts()
    print(
        f"{col}: categories occurring fewer than 10 times:",
        int((counts < 10).sum())
    )

# 8. Amount analysis
print("\n8. AMOUNT ANALYSIS")

print(
    df.groupby("invoice_currency")["total_open_amount"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
    .to_string()
)

print(
    "Zero amounts:",
    (df["total_open_amount"] == 0).sum()
)

print(
    "Negative amounts:",
    (df["total_open_amount"] < 0).sum()
)

print(
    "Infinite amounts:",
    np.isinf(df["total_open_amount"]).sum()
)

# 9. Customer history availability
print("\n9. CUSTOMER HISTORY AUDIT")

customer_counts = df.groupby("cust_number").size()

print("Customers:", len(customer_counts))
print("Single-invoice customers:", (customer_counts == 1).sum())
print("Median invoices per customer:", customer_counts.median())
print("Maximum invoices per customer:", customer_counts.max())

same_day = df.groupby(
    ["cust_number", "posting_date"]
).size()

print("Customer-date groups:", len(same_day))
print("Same-day multiple-invoice groups:", (same_day > 1).sum())
print("Invoices in those groups:", same_day[same_day > 1].sum())

# 10. Historical target availability
print("\n10. CUSTOMER HISTORICAL PAYMENT DATA")

cleared = df[df["clear_date"].notna()].copy()

print("Cleared invoices:", len(cleared))
print("Open invoices:", df["clear_date"].isna().sum())

history = cleared.groupby("cust_number").agg(
    cleared_invoice_count=("late_payment", "count"),
    historical_late_rate=("late_payment", "mean"),
    historical_avg_delay=("payment_delay_days", "mean")
)

print("\nCustomer history summary:")
print(history.describe().round(3).to_string())

# 11. Snapshot and temporal coverage
print("\n11. TEMPORAL DISTRIBUTION")

print(
    df.groupby(df["posting_date"].dt.year)
    .agg(
        invoice_count=("posting_date", "size"),
        customer_count=("cust_number", "nunique"),
        open_invoices=("is_open", "sum")
    )
    .to_string()
)

# 12. Leakage review
print("\n12. LEAKAGE REVIEW")

leakage_cols = [
    "clear_date",
    "payment_delay_days",
    "days_to_clear",
    "late_payment"
]

print("Outcome-related columns:")
print(leakage_cols)

print("\nPrediction-time candidate columns:")
print([
    "business_code",
    "cust_number",
    "posting_date",
    "document_create_date_alt",
    "due_in_date",
    "invoice_currency",
    "document_type",
    "total_open_amount",
    "baseline_create_date",
    "cust_payment_terms"
])

# Save candidate feature audit
features.to_csv(
    OUTPUT_DIR / "date_feature_audit.csv",
    index=False
)

history.to_csv(
    OUTPUT_DIR / "customer_history_audit.csv"
)

print("\n" + "=" * 65)
print("WEEK 3 COMPREHENSIVE AUDIT COMPLETED")
print("Reports saved to:", OUTPUT_DIR)
print("=" * 65)

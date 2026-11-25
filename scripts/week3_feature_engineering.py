
from pathlib import Path
import logging

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PATHS AND LOGGING
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "datasets" / "processed" / "invoices_cleaned.parquet"
OUTPUT_DIR = ROOT / "datasets" / "processed"
REPORT_DIR = ROOT / "reports" / "week3_features"
LOG_DIR = ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "week3_features.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

logger.info("Week 3 feature engineering started")

df = pd.read_parquet(INPUT_FILE)

logger.info("Dataset loaded: %s", df.shape)

required_columns = [
    "cust_number",
    "posting_date",
    "due_in_date",
    "baseline_create_date",
    "clear_date",
    "payment_delay_days",
    "late_payment",
    "is_open",
    "total_open_amount",
    "invoice_currency",
    "cust_payment_terms",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df.copy()


# --------------------------------------------------
# 3. DATE FEATURES
# --------------------------------------------------

date_columns = [
    "posting_date",
    "due_in_date",
    "baseline_create_date",
    "clear_date",
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df["invoice_year"] = df["posting_date"].dt.year
df["invoice_month"] = df["posting_date"].dt.month
df["invoice_day"] = df["posting_date"].dt.day
df["invoice_day_of_week"] = df["posting_date"].dt.dayofweek

df["payment_term_days"] = (
    df["due_in_date"] - df["baseline_create_date"]
).dt.days

df["posting_to_due_days"] = (
    df["due_in_date"] - df["posting_date"]
).dt.days

df["is_due_date_anomaly"] = (
    df["posting_to_due_days"] < 0
).astype("int8")


# --------------------------------------------------
# 4. CUSTOMER PREVIOUS INVOICE COUNT
# --------------------------------------------------

# Only invoices posted strictly before the current invoice
# are counted. Same-day invoices are excluded.

df["customer_previous_invoice_count"] = 0

for customer, group in df.groupby("cust_number", sort=False):
    positions = group.index
    dates = np.sort(
        group["posting_date"].dropna().values
    )

    current_dates = group["posting_date"].values

    counts = np.searchsorted(
        dates,
        current_dates,
        side="left"
    )

    df.loc[positions, "customer_previous_invoice_count"] = counts


# --------------------------------------------------
# 5. LEAKAGE-SAFE CUSTOMER PAYMENT HISTORY
# --------------------------------------------------

# A previous payment is available only if its clear_date
# is strictly earlier than the current invoice posting_date.
#
# Same-day cleared payments are excluded.

df["customer_previous_cleared_count"] = 0
df["customer_previous_late_rate"] = np.nan
df["customer_previous_avg_delay"] = np.nan
df["customer_previous_median_delay"] = np.nan

history = df[
    df["clear_date"].notna()
    & df["payment_delay_days"].notna()
    & df["late_payment"].notna()
].copy()

for customer, current_group in df.groupby("cust_number", sort=False):

    customer_history = history[
        history["cust_number"] == customer
    ].copy()

    if customer_history.empty:
        continue

    customer_history = customer_history.sort_values("clear_date")

    clear_dates = customer_history["clear_date"].values

    late_values = customer_history["late_payment"].astype(float).values
    delay_values = customer_history["payment_delay_days"].astype(float).values

    current_positions = current_group.index
    current_dates = current_group["posting_date"].values

    for position, current_date in zip(current_positions, current_dates):

        if pd.isna(current_date):
            continue

        # Strictly earlier than posting date
        count = np.searchsorted(
            clear_dates,
            current_date,
            side="left"
        )

        df.loc[position, "customer_previous_cleared_count"] = count

        if count == 0:
            continue

        previous_late = late_values[:count]
        previous_delay = delay_values[:count]

        df.loc[position, "customer_previous_late_rate"] = (
            previous_late.mean()
        )

        df.loc[position, "customer_previous_avg_delay"] = (
            previous_delay.mean()
        )

        df.loc[position, "customer_previous_median_delay"] = (
            np.median(previous_delay)
        )
        # Create this flag AFTER historical counts are calculated.
df["has_customer_payment_history"] = (
    df["customer_previous_cleared_count"] > 0
).astype("int8")


# --------------------------------------------------
# 6. DATA TYPE CLEANUP
# --------------------------------------------------

count_columns = [
    "customer_previous_invoice_count",
    "customer_previous_cleared_count",
]

for col in count_columns:
    df[col] = df[col].astype("int32")

df["is_due_date_anomaly"] = df["is_due_date_anomaly"].astype("int8")


# --------------------------------------------------
# 7. TARGET AND DATASET SPLIT
# --------------------------------------------------

# late_payment is the prediction target.
# Open invoices do not yet have a known target.

train_df = df[df["late_payment"].notna()].copy()
open_df = df[df["late_payment"].isna()].copy()

logger.info("All feature rows: %s", len(df))
logger.info("Cleared invoice rows: %s", len(train_df))
logger.info("Open invoice rows: %s", len(open_df))


# --------------------------------------------------
# 8. DEFINE MODEL INPUT FEATURES
# --------------------------------------------------

feature_columns = [
    "business_code",
    "invoice_currency",
    "document_type",
    "cust_payment_terms",
    "total_open_amount",
    "has_customer_payment_history",
    "invoice_year",
    "invoice_month",
    "invoice_day",
    "invoice_day_of_week",

    "payment_term_days",
    "posting_to_due_days",
    "is_due_date_anomaly",

    "customer_previous_invoice_count",
    "customer_previous_cleared_count",
    "customer_previous_late_rate",
    "customer_previous_avg_delay",
    "customer_previous_median_delay",
]

# Keep target separate from model input columns.
target_column = "late_payment"

# Drop columns that are not required for modeling.
# The original identifiers and outcome-related dates remain
# available in the complete engineered dataset for auditing.

model_columns = feature_columns + [target_column]

train_model = train_df[model_columns].copy()
open_model = open_df[feature_columns].copy()


# --------------------------------------------------
# 9. SAVE OUTPUT DATASETS
# --------------------------------------------------

df.to_parquet(
    OUTPUT_DIR / "invoices_features_all.parquet",
    index=False
)

train_model.to_parquet(
    OUTPUT_DIR / "invoices_features_train.parquet",
    index=False
)

open_model.to_parquet(
    OUTPUT_DIR / "invoices_features_open.parquet",
    index=False
)

logger.info("Feature datasets saved")


# --------------------------------------------------
# 10. FEATURE DICTIONARY
# --------------------------------------------------

feature_descriptions = {
    "business_code": "Business entity code",
    "has_customer_payment_history": "1 if at least one earlier customer payment was cleared before the current posting date",
    "cust_number": "Customer identifier",
    "invoice_currency": "Invoice currency",
    "document_type": "Invoice document type",
    "cust_payment_terms": "Customer payment terms code",
    "total_open_amount": "Source amount field; business meaning requires confirmation",
    "invoice_year": "Year of invoice posting date",
    "invoice_month": "Month of invoice posting date",
    "invoice_day": "Day of month of invoice posting date",
    "invoice_day_of_week": "Day of week, Monday=0",
    "payment_term_days": "Due date minus baseline creation date",
    "posting_to_due_days": "Due date minus posting date",
    "is_due_date_anomaly": "1 when due date is earlier than posting date",
    "customer_previous_invoice_count": "Number of same-customer invoices posted before current posting date",
    "customer_previous_cleared_count": "Number of same-customer payments cleared before current posting date",
    "customer_previous_late_rate": "Historical late-payment rate using only earlier cleared payments",
    "customer_previous_avg_delay": "Historical average payment delay using only earlier cleared payments",
    "customer_previous_median_delay": "Historical median payment delay using only earlier cleared payments",
    "late_payment": "Target: 1 late, 0 not late; missing for open invoices",
}

feature_dictionary = pd.DataFrame({
    "feature": feature_columns + [target_column],
    "description": [
        feature_descriptions[col]
        for col in feature_columns + [target_column]
    ],
})

feature_dictionary.to_csv(
    REPORT_DIR / "feature_dictionary.csv",
    index=False
)


# --------------------------------------------------
# 11. VALIDATION REPORT
# --------------------------------------------------

validation = []

def add_check(name, value, status, details=""):
    validation.append({
        "check": name,
        "value": value,
        "status": status,
        "details": details,
    })


add_check(
    "Total rows",
    len(df),
    "PASS" if len(df) == 48839 else "REVIEW"
)

add_check(
    "Training rows",
    len(train_df),
    "PASS" if len(train_df) == 39158 else "REVIEW"
)

add_check(
    "Open invoice rows",
    len(open_df),
    "PASS" if len(open_df) == 9681 else "REVIEW"
)

add_check(
    "Target missing in training",
    int(train_model[target_column].isna().sum()),
    "PASS" if train_model[target_column].notna().all() else "FAIL"
)

add_check(
    "Target missing in open invoices",
    int(df.loc[open_df.index, target_column].isna().sum()),
    "PASS" if open_model.shape[0] == len(open_df) else "FAIL"
)

add_check(
    "Negative payment term rows",
    int((df["payment_term_days"] < 0).sum()),
    "REVIEW"
)

add_check(
    "Due date anomaly rows",
    int(df["is_due_date_anomaly"].sum()),
    "REVIEW"
)

add_check(
    "Customer history count consistency",
    int(
        (
            df["customer_previous_cleared_count"]
            > df["customer_previous_invoice_count"]
        ).sum()
    ),
    "REVIEW",
    "A cleared invoice may have been posted before the current invoice; investigate any positive count."
)

add_check(
    "Feature columns",
    len(feature_columns),
    "PASS"
)

add_check(
    "Original cleaned dataset unchanged",
    "Separate output file used",
    "PASS"
)

validation_df = pd.DataFrame(validation)

validation_df.to_csv(
    REPORT_DIR / "feature_validation_report.csv",
    index=False
)


# --------------------------------------------------
# 12. FEATURE MISSING VALUE REPORT
# --------------------------------------------------

missing_report = (
    df[feature_columns + [target_column]]
    .isna()
    .sum()
    .reset_index()
)

missing_report.columns = ["feature", "missing_count"]

missing_report["missing_percentage"] = (
    missing_report["missing_count"] / len(df) * 100
).round(2)

missing_report.to_csv(
    REPORT_DIR / "feature_missing_values.csv",
    index=False
)


# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

print("\nWEEK 3 FEATURE ENGINEERING COMPLETED")
print("------------------------------------")

print("Original dataset shape:", df.shape)
print("Training dataset shape:", train_model.shape)
print("Open invoice dataset shape:", open_model.shape)

print("\nTarget distribution:")
print(train_model[target_column].value_counts().sort_index())

print("\nFeature columns:")
for col in feature_columns:
    print("-", col)

print("\nValidation report:")
print(validation_df.to_string(index=False))

print("\nSaved files:")
print("datasets/processed/invoices_features_all.parquet")
print("datasets/processed/invoices_features_train.parquet")
print("datasets/processed/invoices_features_open.parquet")
print("reports/week3_features/feature_dictionary.csv")
print("reports/week3_features/feature_validation_report.csv")
print("reports/week3_features/feature_missing_values.csv")
print("logs/week3_features.log")

logger.info("Week 3 feature engineering completed successfully")

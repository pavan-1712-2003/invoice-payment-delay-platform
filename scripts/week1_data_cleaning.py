
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "datasets" / "raw" / "dataset.csv"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "week1_cleaning"
DOCS_DIR = BASE_DIR / "docs"
LOG_DIR = BASE_DIR / "logs"

for folder in [PROCESSED_DIR, REPORT_DIR, DOCS_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

# =====================================================
# 2. LOGGING
# =====================================================

logging.basicConfig(
    filename=LOG_DIR / "week1_cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

# =====================================================
# 3. LOAD DATA
# =====================================================

print("\nLoading dataset...")

df = pd.read_csv(RAW_FILE, low_memory=False)

original_rows = len(df)
original_columns = len(df.columns)

print("Original shape:", df.shape)

logging.info(f"Original shape: {df.shape}")

# =====================================================
# 4. STANDARDIZE COLUMN NAMES
# =====================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

df = df.rename(columns={
    "buisness_year": "business_year",
    "document_type": "document_type",
    "document_create_date.1": "document_create_date_alt",
    "isopen": "is_open"
})

# =====================================================
# 5. REMOVE COMPLETELY EMPTY COLUMNS
# =====================================================

empty_columns = df.columns[df.isna().all()].tolist()

df = df.drop(columns=empty_columns)

print("Removed completely empty columns:", empty_columns)

# =====================================================
# 6. HANDLE EXACT DUPLICATES
# =====================================================

exact_duplicate_mask = df.duplicated(keep="first")

exact_duplicate_rows = df.loc[exact_duplicate_mask].copy()

exact_duplicate_rows.to_csv(
    REPORT_DIR / "exact_duplicate_rows.csv",
    index=False
)

exact_duplicate_count = int(exact_duplicate_mask.sum())

df = df.drop_duplicates(keep="first").copy()

print("Exact duplicate rows removed:", exact_duplicate_count)

# =====================================================
# 7. CLEAN TEXT COLUMNS
# =====================================================

text_columns = [
    "business_code",
    "cust_number",
    "name_customer",
    "invoice_currency",
    "document_type",
    "cust_payment_terms"
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()

# =====================================================
# 8. CONVERT DATE COLUMNS
# =====================================================

def parse_yyyymmdd(series):
    numeric_values = pd.to_numeric(series, errors="coerce")

    integer_values = numeric_values.round().astype("Int64")

    date_strings = integer_values.astype("string")

    return pd.to_datetime(
        date_strings,
        format="%Y%m%d",
        errors="coerce"
    )


# Dates stored as normal date strings
df["clear_date"] = pd.to_datetime(
    df["clear_date"],
    errors="coerce"
)

df["posting_date"] = pd.to_datetime(
    df["posting_date"],
    errors="coerce"
)

# Dates stored as YYYYMMDD numbers
yyyymmdd_columns = [
    "due_in_date",
    "baseline_create_date",
    "document_create_date",
    "document_create_date_alt"
]

for col in yyyymmdd_columns:
    if col in df.columns:
        df[col] = parse_yyyymmdd(df[col])

# =====================================================
# 9. CONVERT NUMERIC COLUMNS
# =====================================================

integer_columns = [
    "business_year",
    "doc_id",
    "posting_id",
    "invoice_id",
    "is_open"
]

for col in integer_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).round().astype("Int64")

df["total_open_amount"] = pd.to_numeric(
    df["total_open_amount"],
    errors="coerce"
)

# =====================================================
# 10. DUPLICATE IDENTIFIER AUDIT
# =====================================================

duplicate_reports = []

for col in ["doc_id", "invoice_id"]:
    if col in df.columns:

        duplicated_mask = (
            df[col].notna()
            & df[col].duplicated(keep=False)
        )

        duplicate_count = int(duplicated_mask.sum())

        duplicate_reports.append({
            "column": col,
            "rows_in_duplicate_groups": duplicate_count,
            "unique_duplicated_values": df.loc[
                duplicated_mask, col
            ].nunique(),
            "missing_values": int(df[col].isna().sum())
        })

pd.DataFrame(duplicate_reports).to_csv(
    REPORT_DIR / "duplicate_report.csv",
    index=False
)

# =====================================================
# 11. DATA VALIDATION
# =====================================================

validation = []

def add_validation(check, count, status):
    validation.append({
        "check": check,
        "count": int(count),
        "status": status
    })


# Missing values
for col in df.columns:
    missing_count = int(df[col].isna().sum())

    add_validation(
        f"Missing values - {col}",
        missing_count,
        "Expected" if col == "clear_date" else
        "Review" if missing_count > 0 else "PASS"
    )


# Invalid date logic
add_validation(
    "Due date before posting date",
    (df["due_in_date"] < df["posting_date"]).sum(),
    "Review"
)

add_validation(
    "Clear date before posting date",
    (df["clear_date"] < df["posting_date"]).sum(),
    "Review"
)

add_validation(
    "Clear date before due date",
    (df["clear_date"] < df["due_in_date"]).sum(),
    "Review"
)

# is_open and clear_date consistency
open_date_mismatch = (
    ((df["is_open"] == 1) & df["clear_date"].notna())
    |
    ((df["is_open"] == 0) & df["clear_date"].isna())
)

add_validation(
    "is_open versus clear_date mismatch",
    open_date_mismatch.sum(),
    "Review"
)

# Negative amount
add_validation(
    "Negative total_open_amount",
    (df["total_open_amount"] < 0).sum(),
    "Review"
)

# Invalid is_open values
add_validation(
    "Invalid is_open values",
    (~df["is_open"].isin([0, 1])).sum(),
    "Review"
)

# Date validity for non-missing source values
date_source_columns = [
    "clear_date",
    "posting_date",
    "due_in_date",
    "baseline_create_date",
    "document_create_date",
    "document_create_date_alt"
]

for col in date_source_columns:
    if col in df.columns:
        # Non-null values after parsing are valid.
        # Missing clear_date values are expected for open invoices.
        pass

pd.DataFrame(validation).to_csv(
    REPORT_DIR / "validation_report.csv",
    index=False
)

# =====================================================
# 12. CREATE PAYMENT TARGETS
# =====================================================

# Only cleared invoices have an observed payment delay.
df["payment_delay_days"] = (
    df["clear_date"] - df["due_in_date"]
).dt.days.astype("Int64")

# Classification target:
# 1 = paid late
# 0 = paid early or on time
# Missing = payment not yet cleared

df["late_payment"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="Int8"
)

cleared_mask = df["clear_date"].notna()

df.loc[cleared_mask, "late_payment"] = (
    df.loc[cleared_mask, "payment_delay_days"] > 0
).astype("int8")

# Days between posting and clearing
df["days_to_clear"] = (
    df["clear_date"] - df["posting_date"]
).dt.days.astype("Int64")

# =====================================================
# 13. OUTLIER DETECTION USING IQR
# =====================================================

numeric_features = [
    "total_open_amount",
    "payment_delay_days",
    "days_to_clear"
]

outlier_results = []

for col in numeric_features:

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (values < lower_bound)
        | (values > upper_bound)
    ).sum()

    outlier_results.append({
        "column": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(outlier_count),
        "outlier_percentage": round(
            outlier_count / len(values) * 100, 2
        )
    })

pd.DataFrame(outlier_results).to_csv(
    REPORT_DIR / "outlier_summary.csv",
    index=False
)

# =====================================================
# 14. CATEGORY SUMMARY
# =====================================================

category_columns = [
    "business_code",
    "invoice_currency",
    "document_type",
    "cust_payment_terms"
]

category_results = []

for col in category_columns:

    if col not in df.columns:
        continue

    counts = df[col].value_counts(dropna=False)

    for category, count in counts.items():

        category_results.append({
            "column": col,
            "category": str(category),
            "count": int(count)
        })

pd.DataFrame(category_results).to_csv(
    REPORT_DIR / "category_summary.csv",
    index=False
)

# =====================================================
# 15. TARGET SUMMARY
# =====================================================

target_summary = pd.DataFrame([
    {
        "metric": "Total invoices",
        "value": len(df)
    },
    {
        "metric": "Cleared invoices",
        "value": int(cleared_mask.sum())
    },
    {
        "metric": "Open invoices",
        "value": int(df["is_open"].eq(1).sum())
    },
    {
        "metric": "Late paid invoices",
        "value": int(df["late_payment"].eq(1).sum())
    },
    {
        "metric": "Early or on-time paid invoices",
        "value": int(df["late_payment"].eq(0).sum())
    },
    {
        "metric": "Missing payment delay targets",
        "value": int(df["payment_delay_days"].isna().sum())
    }
])

target_summary.to_csv(
    REPORT_DIR / "target_summary.csv",
    index=False
)

# =====================================================
# 16. DATA DICTIONARY
# =====================================================

descriptions = {
    "business_code": "Business entity code",
    "cust_number": "Customer identifier",
    "name_customer": "Customer name",
    "clear_date": "Actual invoice clearing date",
    "business_year": "Business year",
    "doc_id": "Document identifier",
    "posting_date": "Invoice posting date",
    "document_create_date": "Document creation date",
    "document_create_date_alt": "Alternate document creation date field",
    "due_in_date": "Invoice due date",
    "invoice_currency": "Invoice currency",
    "document_type": "Accounting document type",
    "posting_id": "Posting identifier",
    "total_open_amount": "Invoice amount field supplied by dataset",
    "baseline_create_date": "Baseline creation date",
    "cust_payment_terms": "Customer payment terms",
    "invoice_id": "Invoice identifier",
    "is_open": "Open invoice indicator",
    "payment_delay_days": "Clear date minus due date",
    "late_payment": "1 if cleared after due date, otherwise 0",
    "days_to_clear": "Clear date minus posting date"
}

data_dictionary = pd.DataFrame({
    "column": df.columns,
    "data_type": [str(df[col].dtype) for col in df.columns],
    "description": [
        descriptions.get(col, "Original dataset field")
        for col in df.columns
    ],
    "missing_values": [
        int(df[col].isna().sum())
        for col in df.columns
    ],
    "unique_values": [
        int(df[col].nunique(dropna=True))
        for col in df.columns
    ]
})

data_dictionary.to_csv(
    DOCS_DIR / "data_dictionary.csv",
    index=False
)

# =====================================================
# 17. CLEANING LOG
# =====================================================

cleaning_log = pd.DataFrame([
    {
        "operation": "Original dataset loaded",
        "rows_affected": original_rows,
        "details": "Loaded raw CSV dataset"
    },
    {
        "operation": "Empty columns removed",
        "rows_affected": 0,
        "details": str(empty_columns)
    },
    {
        "operation": "Exact duplicate rows removed",
        "rows_affected": exact_duplicate_count,
        "details": "Exact full-row duplicates only"
    },
    {
        "operation": "Date conversion",
        "rows_affected": len(df),
        "details": "Converted date columns to datetime"
    },
    {
        "operation": "Numeric conversion",
        "rows_affected": len(df),
        "details": "Converted numeric and identifier columns"
    },
    {
        "operation": "Payment targets created",
        "rows_affected": int(cleared_mask.sum()),
        "details": "Targets created only for cleared invoices"
    },
    {
        "operation": "Outlier detection",
        "rows_affected": len(df),
        "details": "IQR detection; no outliers deleted or capped"
    }
])

cleaning_log.to_csv(
    REPORT_DIR / "cleaning_log.csv",
    index=False
)

# =====================================================
# 18. SAVE CLEANED DATASETS
# =====================================================

csv_output = PROCESSED_DIR / "invoices_cleaned.csv"
parquet_output = PROCESSED_DIR / "invoices_cleaned.parquet"

df.to_csv(csv_output, index=False)

df.to_parquet(
    parquet_output,
    index=False,
    engine="pyarrow"
)

# =====================================================
# 19. FINAL VERIFICATION
# =====================================================

print("\n" + "=" * 55)
print("WEEK 1 DATA CLEANING COMPLETED")
print("=" * 55)

print("Original rows:", original_rows)
print("Cleaned rows:", len(df))
print("Original columns:", original_columns)
print("Cleaned columns:", len(df.columns))
print("Exact duplicates removed:", exact_duplicate_count)

print("\nFinal shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nPayment target distribution:")
print(df["late_payment"].value_counts(dropna=False))

print("\nPayment delay statistics:")
print(df["payment_delay_days"].describe())

print("\nGenerated files:")
print(csv_output)
print(parquet_output)
print(REPORT_DIR)
print(DOCS_DIR / "data_dictionary.csv")

print("\nWeek 1 process finished.")

logging.info("Week 1 cleaning completed successfully.")

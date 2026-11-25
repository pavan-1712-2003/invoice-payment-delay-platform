
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# WEEK 2: EXPLORATORY DATA ANALYSIS
# Invoice Payment Delay & Cash-Flow Intelligence Platform
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "datasets" / "processed" / "invoices_cleaned.parquet"
OUTPUT_DIR = BASE_DIR / "reports" / "week2_eda"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Assumed dataset snapshot date, based on the dataset's
# latest posting date. This is an analytical assumption.
SNAPSHOT_DATE = pd.Timestamp("2020-05-22")

print("=" * 60)
print("WEEK 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\nLoading cleaned dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_parquet(DATA_FILE)

date_columns = [
    "posting_date",
    "document_create_date",
    "due_in_date",
    "clear_date"
]

for column in date_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. DATASET OVERVIEW
# ------------------------------------------------------------

overview = pd.DataFrame({
    "Metric": [
        "Total invoices",
        "Total columns",
        "Cleared invoices",
        "Open invoices",
        "Missing invoice IDs",
        "Duplicate rows",
        "Earliest posting date",
        "Latest posting date",
        "Assumed snapshot date"
    ],
    "Value": [
        len(df),
        len(df.columns),
        int((df["is_open"] == 0).sum()),
        int((df["is_open"] == 1).sum()),
        int(df["invoice_id"].isna().sum()),
        int(df.duplicated().sum()),
        df["posting_date"].min(),
        df["posting_date"].max(),
        SNAPSHOT_DATE
    ]
})

overview.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)

# ------------------------------------------------------------
# 3. PAYMENT DELAY ANALYSIS
# ------------------------------------------------------------

cleared = df[df["clear_date"].notna()].copy()
open_invoices = df[df["clear_date"].isna()].copy()

cleared["payment_delay_days"] = (
    cleared["clear_date"] - cleared["due_in_date"]
).dt.days

delay = cleared["payment_delay_days"].dropna()

delay_summary = pd.DataFrame({
    "Metric": [
        "Total cleared invoices",
        "Average payment delay",
        "Median payment delay",
        "Minimum payment delay",
        "Maximum payment delay",
        "Standard deviation"
    ],
    "Value": [
        len(delay),
        delay.mean(),
        delay.median(),
        delay.min(),
        delay.max(),
        delay.std()
    ]
})

delay_summary.to_csv(
    OUTPUT_DIR / "payment_delay_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 4. PAYMENT BEHAVIOR
# ------------------------------------------------------------

cleared["payment_behavior"] = np.select(
    [
        cleared["payment_delay_days"] < 0,
        cleared["payment_delay_days"] == 0,
        cleared["payment_delay_days"] > 0
    ],
    [
        "Early Payment",
        "On-Time Payment",
        "Late Payment"
    ],
    default="Unknown"
)

behavior = (
    cleared["payment_behavior"]
    .value_counts()
    .rename_axis("payment_behavior")
    .reset_index(name="invoice_count")
)

behavior["percentage"] = (
    behavior["invoice_count"] / len(cleared) * 100
).round(2)

behavior.to_csv(
    OUTPUT_DIR / "payment_behavior.csv",
    index=False
)

# ------------------------------------------------------------
# 5. CUSTOMER ANALYSIS
# ------------------------------------------------------------

customer_analysis = (
    df.groupby("cust_number")
    .agg(
        total_invoices=("cust_number", "size"),
        cleared_invoices=("clear_date", "count"),
        open_invoices=("clear_date", lambda x: x.isna().sum()),
        average_delay_days=("payment_delay_days", "mean"),
        median_delay_days=("payment_delay_days", "median"),
        late_payment_rate=("late_payment", "mean")
    )
    .reset_index()
)

customer_analysis["late_payment_rate"] *= 100

customer_analysis.to_csv(
    OUTPUT_DIR / "customer_analysis.csv",
    index=False
)

# Reliable comparison: customers with at least 10 cleared invoices.
reliable_customers = customer_analysis[
    customer_analysis["cleared_invoices"] >= 10
].copy()

reliable_customers = reliable_customers.sort_values(
    "average_delay_days",
    ascending=False
)

reliable_customers.to_csv(
    OUTPUT_DIR / "customer_analysis_reliable.csv",
    index=False
)

# ------------------------------------------------------------
# 6. PAYMENT TERMS ANALYSIS
# ------------------------------------------------------------

terms_analysis = (
    df.groupby("cust_payment_terms")
    .agg(
        total_invoices=("cust_payment_terms", "size"),
        cleared_invoices=("clear_date", "count"),
        average_delay_days=("payment_delay_days", "mean"),
        median_delay_days=("payment_delay_days", "median"),
        late_payment_rate=("late_payment", "mean")
    )
    .reset_index()
)

terms_analysis["late_payment_rate"] *= 100

terms_analysis.to_csv(
    OUTPUT_DIR / "payment_terms_analysis.csv",
    index=False
)

# Reliable comparison: payment terms with at least 30 cleared invoices.
reliable_terms = terms_analysis[
    terms_analysis["cleared_invoices"] >= 30
].copy()

reliable_terms = reliable_terms.sort_values(
    "average_delay_days",
    ascending=False
)

reliable_terms.to_csv(
    OUTPUT_DIR / "payment_terms_analysis_reliable.csv",
    index=False
)

# ------------------------------------------------------------
# 7. AMOUNT ANALYSIS BY CURRENCY
# ------------------------------------------------------------

amount_analysis = (
    df.groupby("invoice_currency")["total_open_amount"]
    .agg(
        invoice_count="count",
        average_amount="mean",
        median_amount="median",
        minimum_amount="min",
        maximum_amount="max",
        total_amount="sum"
    )
    .reset_index()
)

amount_analysis.to_csv(
    OUTPUT_DIR / "amount_analysis.csv",
    index=False
)

# Amount distribution summary
amount_distribution = df["total_open_amount"].describe().reset_index()
amount_distribution.columns = ["Metric", "Value"]

amount_distribution.to_csv(
    OUTPUT_DIR / "amount_distribution.csv",
    index=False
)

# ------------------------------------------------------------
# 8. MONTHLY TRENDS
# ------------------------------------------------------------

created = (
    df.groupby(df["posting_date"].dt.to_period("M"))
    .size()
    .rename("invoices_created")
)

cleared_monthly = (
    cleared.groupby(cleared["clear_date"].dt.to_period("M"))
    .size()
    .rename("invoices_cleared")
)

monthly = pd.concat(
    [created, cleared_monthly],
    axis=1
).fillna(0)

monthly.index = monthly.index.astype(str)
monthly.index.name = "month"

monthly = monthly.reset_index()

monthly.to_csv(
    OUTPUT_DIR / "monthly_trends.csv",
    index=False
)

# ------------------------------------------------------------
# 9. OPEN INVOICE DUE STATUS
# ------------------------------------------------------------

open_invoices["days_until_due_at_snapshot"] = (
    open_invoices["due_in_date"] - SNAPSHOT_DATE
).dt.days

open_invoices["due_status_at_snapshot"] = np.select(
    [
        open_invoices["days_until_due_at_snapshot"] < -30,
        open_invoices["days_until_due_at_snapshot"].between(-30, -1),
        open_invoices["days_until_due_at_snapshot"] == 0,
        open_invoices["days_until_due_at_snapshot"].between(1, 30),
        open_invoices["days_until_due_at_snapshot"].between(31, 60),
        open_invoices["days_until_due_at_snapshot"] > 60
    ],
    [
        "Overdue >30 days",
        "Overdue 1-30 days",
        "Due on snapshot date",
        "Due in 1-30 days",
        "Due in 31-60 days",
        "Due after 60 days"
    ],
    default="Unknown"
)

due_status = (
    open_invoices["due_status_at_snapshot"]
    .value_counts()
    .rename_axis("due_status")
    .reset_index(name="invoice_count")
)

due_status["percentage"] = (
    due_status["invoice_count"] / len(open_invoices) * 100
).round(2)

due_status.to_csv(
    OUTPUT_DIR / "open_invoice_due_status.csv",
    index=False
)

due_summary = pd.DataFrame({
    "Metric": [
        "Assumed snapshot date",
        "Total open invoices",
        "Not yet due",
        "Due on snapshot date",
        "Past due",
        "Average days until due",
        "Median days until due"
    ],
    "Value": [
        SNAPSHOT_DATE,
        len(open_invoices),
        int((open_invoices["days_until_due_at_snapshot"] > 0).sum()),
        int((open_invoices["days_until_due_at_snapshot"] == 0).sum()),
        int((open_invoices["days_until_due_at_snapshot"] < 0).sum()),
        open_invoices["days_until_due_at_snapshot"].mean(),
        open_invoices["days_until_due_at_snapshot"].median()
    ]
})

due_summary.to_csv(
    OUTPUT_DIR / "open_invoice_due_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 10. CORRELATION ANALYSIS
# ------------------------------------------------------------

correlation_columns = [
    "business_year",
    "total_open_amount",
    "is_open",
    "payment_delay_days",
    "late_payment"
]

correlation_data = df[correlation_columns].corr(numeric_only=True)

correlation_data.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

# This is exploratory only.
# Do not use this matrix as a feature-selection result.
# payment_delay_days and late_payment are outcome-related.

# ------------------------------------------------------------
# 11. CHARTS
# ------------------------------------------------------------

print("\nGenerating charts...")

# Payment delay distribution
plt.figure(figsize=(10, 6))
plt.hist(delay, bins=50, edgecolor="black")
plt.title("Payment Delay Distribution")
plt.xlabel("Payment Delay (Days)")
plt.ylabel("Invoice Count")
plt.tight_layout()
plt.savefig(CHART_DIR / "payment_delay_distribution.png")
plt.close()

# Payment behavior
behavior_plot = behavior.set_index("payment_behavior")

plt.figure(figsize=(8, 5))
plt.bar(
    behavior_plot.index,
    behavior_plot["invoice_count"]
)
plt.title("Invoice Payment Behavior")
plt.xlabel("Payment Behavior")
plt.ylabel("Invoice Count")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(CHART_DIR / "payment_behavior.png")
plt.close()

# Monthly trends
plt.figure(figsize=(12, 6))
plt.plot(
    monthly["month"],
    monthly["invoices_created"],
    marker="o",
    label="Invoices Created"
)
plt.plot(
    monthly["month"],
    monthly["invoices_cleared"],
    marker="o",
    label="Invoices Cleared"
)
plt.title("Monthly Invoice Trends")
plt.xlabel("Month")
plt.ylabel("Invoice Count")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(CHART_DIR / "monthly_invoice_trends.png")
plt.close()

# Payment delay by currency
currency_groups = [
    group["payment_delay_days"].dropna().values
    for _, group in cleared.groupby("invoice_currency")
]

currency_labels = [
    name for name, _ in cleared.groupby("invoice_currency")
]

plt.figure(figsize=(8, 5))
plt.boxplot(
    currency_groups,
    tick_labels=currency_labels
)
plt.title("Payment Delay by Currency")
plt.xlabel("Invoice Currency")
plt.ylabel("Payment Delay (Days)")
plt.tight_layout()
plt.savefig(CHART_DIR / "payment_delay_by_currency.png")
plt.close()

# Correlation heatmap using Matplotlib
plt.figure(figsize=(9, 7))

plt.imshow(
    correlation_data,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_data.columns)),
    correlation_data.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_data.index)),
    correlation_data.index
)

for i in range(len(correlation_data.index)):
    for j in range(len(correlation_data.columns)):
        value = correlation_data.iloc[i, j]

        if pd.notna(value):
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center"
            )

plt.title("Exploratory Correlation Matrix")
plt.tight_layout()
plt.savefig(CHART_DIR / "correlation_heatmap.png")
plt.close()

# ------------------------------------------------------------
# 12. WEEK 2 SUMMARY
# ------------------------------------------------------------

late_count = int((cleared["payment_delay_days"] > 0).sum())
early_count = int((cleared["payment_delay_days"] < 0).sum())
ontime_count = int((cleared["payment_delay_days"] == 0).sum())

past_due_count = int(
    (open_invoices["days_until_due_at_snapshot"] < 0).sum()
)

summary = f"""
WEEK 2 - EXPLORATORY DATA ANALYSIS SUMMARY

DATASET
-------
Total invoices: {len(df)}
Total columns: {len(df.columns)}
Cleared invoices: {len(cleared)}
Open invoices: {len(open_invoices)}

PAYMENT BEHAVIOR
----------------
Early payments: {early_count}
On-time payments: {ontime_count}
Late payments: {late_count}

Late payment percentage among cleared invoices:
{late_count / len(cleared) * 100:.2f}%

PAYMENT DELAY
-------------
Average delay: {delay.mean():.2f} days
Median delay: {delay.median():.2f} days
Minimum delay: {delay.min()} days
Maximum delay: {delay.max()} days

OPEN INVOICE ANALYSIS
---------------------
Assumed snapshot date: {SNAPSHOT_DATE.date()}
Past-due open invoices: {past_due_count}
Not-yet-due open invoices:
{int((open_invoices['days_until_due_at_snapshot'] > 0).sum())}
Open invoices due on snapshot date:
{int((open_invoices['days_until_due_at_snapshot'] == 0).sum())}

DATASET LIMITATIONS
-------------------
1. Open invoices are concentrated in the latest dataset period.
2. The assumed snapshot date requires source confirmation.
3. total_open_amount meaning is unverified.
4. Amount comparisons across currencies require conversion.
5. Payment delay and late_payment are outcome-related variables.
6. Monthly created and cleared counts are not invoice cohort conversion rates.
7. Correlation analysis is exploratory, not feature selection.
8. Outliers and documented date anomalies were retained.

MODELING CAUTION
----------------
Do not use clear_date, payment_delay_days, or late_payment
as invoice-time prediction input features.

Do not use is_open as an invoice-time prediction feature.
Define the prediction date and available information first.

Week 2 EDA completed.
"""

with open(
    OUTPUT_DIR / "week2_eda_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

# ------------------------------------------------------------
# 13. COMPLETION
# ------------------------------------------------------------

print("\nCharts generated successfully.")

print("\nPAYMENT BEHAVIOR")
print(behavior.to_string(index=False))

print("\nOPEN INVOICE DUE STATUS")
print(due_status.to_string(index=False))

print("\nWEEK 2 EDA COMPLETED")
print("Dataset shape:", df.shape)
print("Cleared invoices:", len(cleared))
print("Open invoices:", len(open_invoices))
print("Average payment delay:", round(delay.mean(), 2), "days")
print("Median payment delay:", round(delay.median(), 2), "days")
print("Past-due open invoices:", past_due_count)
print("Output folder:", OUTPUT_DIR)

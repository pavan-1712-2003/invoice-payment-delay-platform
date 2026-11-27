import json
from pathlib import Path

cells = []

def add_markdown(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    })


add_markdown("""# Week 4 - Invoice Late Payment Classification

## Project objective
Build a classification system that estimates the probability of late payment for an invoice.

**Prediction point:** Invoice posting date.

**Target:** `late_payment` (1 = late, 0 = on-time or early).

This notebook documents data preparation, chronological evaluation, model comparison, cross-validation, probability calibration and invoice risk predictions.

The model is evaluated on historical settled invoices. Open invoices have unknown outcomes and are used only for inference.
""")

add_markdown("""## 1. Methodology and data controls

The modelling workflow follows chronological partitions:

| Partition | Period | Purpose |
|---|---|---|
| Training | Through August 31, 2019 | Model fitting |
| Validation | September-November 2019 | Model and threshold comparison |
| Calibration | December 2019-January 2020 | Sigmoid calibration |
| Final test | February 1-27, 2020 | Independent evaluation |

The model uses 17 approved features. `total_open_amount` is excluded because its business meaning has not been confirmed.

Outcome fields, customer identifiers and invoice identifiers are not model inputs.

Customer historical features must represent only information available before the current invoice's posting date.
""")

add_code("""from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, Image

DATA_DIR = Path("datasets/processed")
REPORT_DIR = Path("reports/week4_modeling")
MODEL_DIR = Path("models")

print("Project directory:", Path.cwd())
print("Report directory exists:", REPORT_DIR.exists())
""")

add_markdown("## 2. Load data and define approved features")

add_code("""data_path = DATA_DIR / "invoices_features_all.parquet"

df = pd.read_parquet(data_path)
df["posting_date"] = pd.to_datetime(df["posting_date"])

features = [
    "business_code",
    "invoice_currency",
    "document_type",
    "cust_payment_terms",
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
    "customer_previous_median_delay"
]

forbidden_features = [
    "late_payment",
    "clear_date",
    "payment_delay_days",
    "days_to_clear",
    "posting_date",
    "total_open_amount",
    "invoice_id",
    "doc_id",
    "cust_number",
    "is_open"
]

missing_features = [col for col in features if col not in df.columns]
assert not missing_features, f"Missing features: {missing_features}"
assert not set(features).intersection(forbidden_features)

print("Full dataset shape:", df.shape)
print("Approved feature count:", len(features))
print("Missing approved features:", missing_features)
print("Settled invoices:", int(df["late_payment"].notna().sum()))
print("Open invoices:", int(df["late_payment"].isna().sum()))

display(df[features].dtypes.rename("data_type").to_frame())
""")

add_markdown("## 3. Chronological data partitions")

add_code("""labelled = df[df["late_payment"].notna()].copy()
labelled["late_payment"] = labelled["late_payment"].astype(int)

train = labelled[labelled["posting_date"] <= "2019-08-31"].copy()

validation = labelled[
    (labelled["posting_date"] >= "2019-09-01") &
    (labelled["posting_date"] <= "2019-11-30")
].copy()

calibration = labelled[
    (labelled["posting_date"] >= "2019-12-01") &
    (labelled["posting_date"] <= "2020-01-31")
].copy()

test = labelled[
    (labelled["posting_date"] >= "2020-02-01") &
    (labelled["posting_date"] <= "2020-02-27")
].copy()

partition_summary = pd.DataFrame([
    ["Training", len(train), train["posting_date"].min(), train["posting_date"].max()],
    ["Validation", len(validation), validation["posting_date"].min(), validation["posting_date"].max()],
    ["Calibration", len(calibration), calibration["posting_date"].min(), calibration["posting_date"].max()],
    ["Final test", len(test), test["posting_date"].min(), test["posting_date"].max()]
], columns=["partition", "rows", "start_date", "end_date"])

display(partition_summary)

print("Training target counts:")
display(train["late_payment"].value_counts().sort_index().to_frame("invoices"))

assert len(train) == 23716
assert len(validation) == 8679
assert len(calibration) == 4404
assert len(test) == 2359
""")

add_markdown("""## 4. Initial model comparison

The following table contains the recorded September-November validation results from the model experiments.

The imbalance handling differs between experiments: Random Forest used balanced class weights, weighted XGBoost used `scale_pos_weight`, and Logistic Regression and unweighted XGBoost did not use class weighting. Keep this limitation in mind when comparing them.
""")

add_code("""comparison = pd.read_csv(REPORT_DIR / "model_comparison.csv")
display(comparison.sort_values("f1", ascending=False))
""")

add_markdown("""## 5. Five-fold expanding-window time-aware cross-validation

The folds split unique posting dates, rather than invoice rows. Invoices posted on the same date remain in one partition.

Preprocessing is fitted separately within each fold. The classification threshold for this comparison is 0.50.
""")

add_code("""cv_summary = pd.read_csv(REPORT_DIR / "time_cv_summary.csv")
cv_folds = pd.read_csv(REPORT_DIR / "time_cv_fold_results.csv")

display(cv_summary)
display(cv_folds.head(10))

print("Fold count:", cv_folds["fold"].nunique())
print("Unique models:", cv_folds["model"].nunique())
""")

add_markdown("## 6. Random Forest threshold analysis")

add_code("""thresholds = pd.read_csv(REPORT_DIR / "threshold_analysis.csv")
display(thresholds)

best_threshold_row = thresholds.loc[thresholds["f1"].idxmax()]
print("Best tested threshold:", best_threshold_row["threshold"])
print("Validation F1:", best_threshold_row["f1"])
print("Validation precision:", best_threshold_row["precision"])
print("Validation recall:", best_threshold_row["recall"])
""")

add_markdown("""## 7. Probability calibration

The Random Forest was fitted on the training partition. A sigmoid calibrator was fitted using the separate December 2019-January 2020 calibration partition.

The February 2020 final test partition was not used to fit the model or calibrator.
""")

add_code("""calibration_metrics = pd.read_csv(
    REPORT_DIR / "calibration_test_metrics.csv"
)
display(calibration_metrics)

brier_change = (
    calibration_metrics.iloc[0]["brier_score"] -
    calibration_metrics.iloc[1]["brier_score"]
)
logloss_change = (
    calibration_metrics.iloc[0]["log_loss"] -
    calibration_metrics.iloc[1]["log_loss"]
)

print("Brier score improvement:", round(brier_change, 6))
print("Log loss improvement:", round(logloss_change, 6))

display(Image(filename=str(REPORT_DIR / "calibration_curve_test.png")))
""")

add_markdown("## 8. Final February 2020 test evaluation")

add_code("""final_metrics = pd.read_csv(
    REPORT_DIR / "final_test_classification_metrics.csv"
)
display(final_metrics)

confusion = pd.read_csv(
    REPORT_DIR / "final_test_confusion_matrix.csv",
    index_col=0
)
display(confusion)

error_analysis = pd.read_csv(
    REPORT_DIR / "final_test_error_analysis.csv"
)
display(error_analysis)

classification_report = pd.read_csv(
    REPORT_DIR / "final_test_classification_report.csv",
    index_col=0
)
display(classification_report)
""")

add_markdown("## 9. Final test performance plots")

add_code("""display(Image(filename=str(REPORT_DIR / "final_test_roc_curve.png")))
display(Image(filename=str(REPORT_DIR / "final_test_precision_recall_curve.png")))
""")

add_markdown("""## 10. Anonymous example predictions

These are example records from the historical final test set. Identifiers and customer details are not displayed.
""")

add_code("""examples = pd.read_csv(
    REPORT_DIR / "anonymous_example_predictions.csv"
)
display(examples.head(25))
""")

add_markdown("""## 11. Predict late-payment risk for open invoices

The saved sigmoid-calibrated Random Forest is applied to the open invoice feature set.

These invoices have no known outcome labels. The resulting probabilities are predictions, not verified outcomes.

The alert threshold below is the calibrated equivalent of the previous raw Random Forest threshold of 0.40. It is provisional, not an independently optimized business threshold.
""")

add_code("""open_path = DATA_DIR / "invoices_features_open.parquet"
model_path = MODEL_DIR / "random_forest_calibrated.joblib"

open_df = pd.read_parquet(open_path)

missing_open_features = [
    col for col in features if col not in open_df.columns
]
assert not missing_open_features, missing_open_features

calibrated_model = joblib.load(model_path)

X_open = open_df[features]
open_probabilities = calibrated_model.predict_proba(X_open)[:, 1]

CALIBRATED_THRESHOLD = 0.3491532574508104

open_predictions = pd.DataFrame({
    "prediction_record": np.arange(1, len(open_df) + 1),
    "predicted_late_probability_percent": np.round(
        open_probabilities * 100, 2
    ),
    "late_payment_alert": np.where(
        open_probabilities >= CALIBRATED_THRESHOLD,
        "Alert",
        "No alert"
    )
})

output_path = REPORT_DIR / "open_invoice_risk_predictions.csv"
open_predictions.to_csv(output_path, index=False)

print("Open invoices scored:", len(open_predictions))
print("Prediction file:", output_path)
print("Mean predicted late probability:",
      round(open_probabilities.mean() * 100, 2), "%")
print("\\nAlert counts:")
display(open_predictions["late_payment_alert"].value_counts().to_frame("invoices"))
display(open_predictions.head(20))
""")

add_markdown("""## 12. Limitations and conclusions

- Evaluation metrics are based on historical settled invoices. Open invoices do not have known labels, so performance on them cannot currently be verified.
- The historical data covers a limited time period. Performance may change as payment behaviour changes.
- The business definition of `total_open_amount` remains unconfirmed; this field was excluded.
- The alert threshold is provisional because the business costs of false positives and false negatives have not been quantified.
- Model comparisons used different imbalance treatments, which limits strict comparability.
- The five-fold time-aware cross-validation provides a more robust chronological comparison than a single random split.
- Random Forest was selected based on its mean F1 and recall across the five folds. XGBoost had stronger mean ROC-AUC and PR-AUC.

**Current candidate:** Sigmoid-calibrated Random Forest.

**Final historical test:** 2,359 invoices from February 1-27, 2020.

**Classification threshold:** 0.34915 on the calibrated probability scale, pending business validation.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = Path("notebooks/week4_classification.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(notebook, file, indent=2)

print("Notebook created:", output_path)
print("Notebook cells:", len(cells))
print("File size:", output_path.stat().st_size, "bytes")

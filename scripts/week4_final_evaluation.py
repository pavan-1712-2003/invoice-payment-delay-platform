from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


# PATHS
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/week4_modeling")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = "datasets/processed/invoices_features_all.parquet"

RAW_THRESHOLD = 0.40


# LOAD DATA
print("Loading data and saved models...")

df = pd.read_parquet(DATA_PATH)
df["posting_date"] = pd.to_datetime(df["posting_date"])
df = df[df["late_payment"].notna()].copy()


# APPROVED FEATURES
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


# CHRONOLOGICAL DATA
validation = df[
    (df["posting_date"] >= "2019-09-01") &
    (df["posting_date"] <= "2019-11-30")
].copy()

test = df[
    (df["posting_date"] >= "2020-02-01") &
    (df["posting_date"] <= "2020-02-27")
].copy()

X_val = validation[features]
y_val = validation["late_payment"].astype(int)

X_test = test[features]
y_test = test["late_payment"].astype(int)

print("Validation rows:", len(X_val))
print("Final test rows:", len(X_test))


# LOAD SAVED MODELS
raw_model = joblib.load(
    MODEL_DIR / "random_forest_uncalibrated.joblib"
)

calibrated_model = joblib.load(
    MODEL_DIR / "random_forest_calibrated.joblib"
)


# VALIDATION PROBABILITIES
raw_val_prob = raw_model.predict_proba(X_val)[:, 1]
cal_val_prob = calibrated_model.predict_proba(X_val)[:, 1]


# TRANSLATE OLD THRESHOLD TO CALIBRATED SCALE
order = np.argsort(raw_val_prob)

sorted_raw = raw_val_prob[order]
sorted_calibrated = cal_val_prob[order]

if np.any(np.diff(sorted_calibrated) < -1e-8):
    raise ValueError(
        "Calibrated probability mapping is not monotonic. "
        "Review threshold mapping before evaluation."
    )

calibrated_threshold = float(
    np.interp(
        RAW_THRESHOLD,
        sorted_raw,
        sorted_calibrated
    )
)

print("\nTHRESHOLD")
print("Previous raw threshold:", RAW_THRESHOLD)
print("Equivalent calibrated threshold:", calibrated_threshold)


# FINAL TEST PROBABILITIES
raw_test_prob = raw_model.predict_proba(X_test)[:, 1]
cal_test_prob = calibrated_model.predict_proba(X_test)[:, 1]

raw_predictions = (raw_test_prob >= RAW_THRESHOLD).astype(int)

cal_predictions = (
    cal_test_prob >= calibrated_threshold
).astype(int)


# CLASSIFICATION METRICS
def calculate_metrics(y_true, y_pred, probabilities, name):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    }


results = [
    calculate_metrics(
        y_test,
        raw_predictions,
        raw_test_prob,
        "Uncalibrated Random Forest"
    ),
    calculate_metrics(
        y_test,
        cal_predictions,
        cal_test_prob,
        "Sigmoid Calibrated Random Forest"
    )
]

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "final_test_classification_metrics.csv",
    index=False
)

print("\nFINAL TEST CLASSIFICATION RESULTS")
print(results_df.to_string(index=False))


# CONFUSION MATRIX
cm = confusion_matrix(
    y_test, cal_predictions, labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual on-time", "Actual late"],
    columns=["Predicted on-time", "Predicted late"]
)

cm_df.to_csv(
    REPORT_DIR / "final_test_confusion_matrix.csv"
)

print("\nCALIBRATED MODEL CONFUSION MATRIX")
print(cm_df)


# CLASSIFICATION REPORT
report = classification_report(
    y_test,
    cal_predictions,
    labels=[0, 1],
    target_names=["On-time", "Late"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    REPORT_DIR / "final_test_classification_report.csv"
)


# ERROR ANALYSIS
tn, fp, fn, tp = cm.ravel()

error_summary = pd.DataFrame([
    {"error_type": "False positives", "count": fp,
     "meaning": "On-time invoices incorrectly flagged as late"},
    {"error_type": "False negatives", "count": fn,
     "meaning": "Late invoices missed by the model"},
    {"error_type": "True positives", "count": tp,
     "meaning": "Late invoices correctly identified"},
    {"error_type": "True negatives", "count": tn,
     "meaning": "On-time invoices correctly identified"}
])

error_summary.to_csv(
    REPORT_DIR / "final_test_error_analysis.csv",
    index=False
)


# ANONYMOUS EXAMPLE PREDICTIONS
examples = pd.DataFrame({
    "test_record_number": range(1, len(test) + 1),
    "actual_late_payment": y_test.to_numpy(),
    "predicted_late_probability_percent": (
        cal_test_prob * 100
    ).round(2),
    "late_payment_alert": np.where(
        cal_predictions == 1, "Alert", "No alert"
    )
})

examples.head(25).to_csv(
    REPORT_DIR / "anonymous_example_predictions.csv",
    index=False
)


# ROC CURVE
fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    raw_test_prob,
    name="Uncalibrated",
    ax=ax
)

RocCurveDisplay.from_predictions(
    y_test,
    cal_test_prob,
    name="Calibrated",
    ax=ax
)

ax.set_title("Random Forest ROC Curve - Final Test")
fig.tight_layout()

fig.savefig(
    REPORT_DIR / "final_test_roc_curve.png",
    dpi=150
)

plt.close(fig)


# PRECISION-RECALL CURVE
fig, ax = plt.subplots(figsize=(8, 6))

PrecisionRecallDisplay.from_predictions(
    y_test,
    raw_test_prob,
    name="Uncalibrated",
    ax=ax
)

PrecisionRecallDisplay.from_predictions(
    y_test,
    cal_test_prob,
    name="Calibrated",
    ax=ax
)

ax.set_title("Random Forest Precision-Recall Curve - Final Test")
fig.tight_layout()

fig.savefig(
    REPORT_DIR / "final_test_precision_recall_curve.png",
    dpi=150
)

plt.close(fig)


# SAVE EVALUATION METADATA
metadata = {
    "raw_threshold": RAW_THRESHOLD,
    "calibrated_threshold": calibrated_threshold,
    "validation_period": "2019-09-01 to 2019-11-30",
    "final_test_period": "2020-02-01 to 2020-02-27",
    "validation_rows": len(validation),
    "final_test_rows": len(test),
    "features": features,
    "example_predictions": "First 25 final-test records; no customer identifiers"
}

with open(
    REPORT_DIR / "final_evaluation_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)

print("\nFinal classification evaluation completed.")
print("Reports saved in:", REPORT_DIR)

from pathlib import Path
import json
import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.frozen import FrozenEstimator


# PATHS
DATA_PATH = Path("datasets/processed/invoices_features_all.parquet")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/week4_modeling")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename="logs/week4_calibration.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# LOAD DATA
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)

df["posting_date"] = pd.to_datetime(df["posting_date"])
df = df[df["late_payment"].notna()].copy()


# CHRONOLOGICAL SPLITS
train = df[df["posting_date"] <= "2019-08-31"].copy()

calibration = df[
    (df["posting_date"] >= "2019-12-01") &
    (df["posting_date"] <= "2020-01-31")
].copy()

test = df[
    (df["posting_date"] >= "2020-02-01") &
    (df["posting_date"] <= "2020-02-27")
].copy()

print("\nDATA SPLITS")
print("Training:", len(train))
print("Calibration:", len(calibration))
print("Final test:", len(test))


# APPROVED FEATURES
feature_columns = [
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

missing_features = [
    col for col in feature_columns
    if col not in df.columns
]

if missing_features:
    raise ValueError(f"Missing model features: {missing_features}")

if train.empty or calibration.empty or test.empty:
    raise ValueError("One or more chronological data splits are empty.")

if y_classes := sorted(train["late_payment"].astype(int).unique()):
    if y_classes != [0, 1]:
        raise ValueError(f"Unexpected training target classes: {y_classes}")


X_train = train[feature_columns]
y_train = train["late_payment"].astype(int)

X_cal = calibration[feature_columns]
y_cal = calibration["late_payment"].astype(int)

X_test = test[feature_columns]
y_test = test["late_payment"].astype(int)


# PREPROCESSING
categorical = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numeric = [
    col for col in feature_columns
    if col not in categorical
]

preprocessor = ColumnTransformer([
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10
            ))
        ]),
        categorical
    ),
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numeric
    )
])


# RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=150,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=2
)

base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf)
])


# TRAIN BASE MODEL
print("\nTraining Random Forest...")
base_pipeline.fit(X_train, y_train)
print("Random Forest training completed.")


# FIT SIGMOID CALIBRATION
print("\nFitting sigmoid calibration...")

calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(base_pipeline),
    method="sigmoid"
)

calibrated_model.fit(X_cal, y_cal)


# EVALUATE ON UNTOUCHED FINAL TEST DATA
print("\nEvaluating on February 2020 test data...")

raw_prob = base_pipeline.predict_proba(X_test)[:, 1]
calibrated_prob = calibrated_model.predict_proba(X_test)[:, 1]

results = []

for name, probabilities in [
    ("Uncalibrated Random Forest", raw_prob),
    ("Sigmoid Calibrated Random Forest", calibrated_prob)
]:
    results.append({
        "model": name,
        "brier_score": brier_score_loss(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1]),
        "test_rows": len(y_test)
    })

metrics_df = pd.DataFrame(results)

metrics_df.to_csv(
    REPORT_DIR / "calibration_test_metrics.csv",
    index=False
)

print("\nFINAL TEST CALIBRATION METRICS")
print(metrics_df.to_string(index=False))


# CALIBRATION CURVE
fig, ax = plt.subplots(figsize=(8, 6))

CalibrationDisplay.from_predictions(
    y_test,
    raw_prob,
    n_bins=10,
    strategy="quantile",
    name="Uncalibrated",
    ax=ax
)

CalibrationDisplay.from_predictions(
    y_test,
    calibrated_prob,
    n_bins=10,
    strategy="quantile",
    name="Sigmoid calibrated",
    ax=ax
)

ax.set_title("Random Forest Calibration - February 2020 Test")
fig.tight_layout()

fig.savefig(
    REPORT_DIR / "calibration_curve_test.png",
    dpi=150
)

plt.close(fig)


# SAVE MODELS
joblib.dump(
    base_pipeline,
    MODEL_DIR / "random_forest_uncalibrated.joblib"
)

joblib.dump(
    calibrated_model,
    MODEL_DIR / "random_forest_calibrated.joblib"
)


# SAVE METADATA
metadata = {
    "model": "RandomForestClassifier",
    "n_estimators": 150,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "calibration": "sigmoid",
    "training_period": "Through 2019-08-31",
    "calibration_period": "2019-12-01 to 2020-01-31",
    "final_test_period": "2020-02-01 to 2020-02-27",
    "features": feature_columns,
    "total_open_amount_excluded": True
}

with open(MODEL_DIR / "calibration_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

logger.info("Calibration and independent test evaluation completed.")

print("\nCalibration completed successfully.")
print("Models saved in:", MODEL_DIR)
print("Reports saved in:", REPORT_DIR)
print("February test data was not used during training or calibration.")


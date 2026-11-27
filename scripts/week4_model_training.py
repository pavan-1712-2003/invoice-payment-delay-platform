from pathlib import Path
import logging

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# PATHS
DATA_PATH = "datasets/processed/invoices_features_all.parquet"
REPORT_DIR = Path("reports/week4_modeling")
LOG_DIR = Path("logs")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "week4_model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# APPROVED FEATURES
FEATURES = [
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

CATEGORICAL = [
    "business_code",
    "invoice_currency",
    "document_type",
    "cust_payment_terms"
]

THRESHOLDS = np.arange(0.30, 0.71, 0.05)


# LOAD DATA
print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)

df["posting_date"] = pd.to_datetime(df["posting_date"])

df = df[df["late_payment"].notna()].copy()
df["late_payment"] = df["late_payment"].astype(int)

missing = [col for col in FEATURES if col not in df.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")


# CHRONOLOGICAL SPLIT
train = df[df["posting_date"] <= "2019-08-31"].copy()

validation = df[
    (df["posting_date"] >= "2019-09-01") &
    (df["posting_date"] <= "2019-11-30")
].copy()

if train.empty or validation.empty:
    raise ValueError("Training or validation partition is empty.")

X_train = train[FEATURES]
y_train = train["late_payment"]

X_val = validation[FEATURES]
y_val = validation["late_payment"]

print("Training rows:", len(train))
print("Validation rows:", len(validation))

print("Training target distribution:")
print(y_train.value_counts().sort_index())


# PREPROCESSING
NUMERIC = [col for col in FEATURES if col not in CATEGORICAL]


def create_preprocessor(scale_numeric=False):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    return ColumnTransformer([
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10
                ))
            ]),
            CATEGORICAL
        ),
        (
            "numeric",
            Pipeline(numeric_steps),
            NUMERIC
        )
    ])


# MODEL CONFIGURATIONS
negatives = int((y_train == 0).sum())
positives = int((y_train == 1).sum())

scale_pos_weight = negatives / positives

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        True
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=2
        ),
        False
    ),

    "XGBoost unweighted": (
        XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=1,
            random_state=42,
            n_jobs=2
        ),
        False
    ),

    "XGBoost weighted": (
        XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=2
        ),
        False
    )
}


# MAJORITY BASELINE
print("\nEvaluating majority baseline...")

majority_class = int(y_train.mode().iloc[0])

majority_predictions = np.full(
    len(y_val),
    majority_class
)

results = [{
    "model": "Majority baseline",
    "accuracy": accuracy_score(y_val, majority_predictions),
    "precision": precision_score(
        y_val, majority_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, majority_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, majority_predictions, zero_division=0
    ),
    "roc_auc": np.nan,
    "pr_auc": np.nan,
    "imbalance_method": "None"
}]


# TRAIN AND EVALUATE MODELS
rf_probabilities = None

for name, (model, scale_numeric) in models.items():

    print(f"\nTraining {name}...")
    logger.info("Training %s", name)

    preprocessor = create_preprocessor(scale_numeric)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "pr_auc": average_precision_score(y_val, probabilities),
        "imbalance_method": (
            "class_weight=balanced" if name == "Random Forest"
            else f"scale_pos_weight={scale_pos_weight:.4f}"
            if name == "XGBoost weighted"
            else "None"
        )
    })

    if name == "Random Forest":
        rf_probabilities = probabilities.copy()

    print(
        f"Accuracy: {results[-1]['accuracy']:.4f} | "
        f"Precision: {results[-1]['precision']:.4f} | "
        f"Recall: {results[-1]['recall']:.4f} | "
        f"F1: {results[-1]['f1']:.4f}"
    )

    logger.info("%s evaluation completed", name)


# SAVE MODEL COMPARISON
comparison = pd.DataFrame(results)

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))


# RANDOM FOREST THRESHOLD ANALYSIS
if rf_probabilities is None:
    raise RuntimeError("Random Forest probabilities not generated.")

threshold_results = []

for threshold in THRESHOLDS:

    predictions = (
        rf_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    REPORT_DIR / "threshold_analysis.csv",
    index=False
)

print("\nRANDOM FOREST THRESHOLD ANALYSIS")
print(threshold_df.to_string(index=False))


# SAVE RUN METADATA
metadata = {
    "training_period": "Through 2019-08-31",
    "validation_period": "2019-09-01 to 2019-11-30",
    "training_rows": len(train),
    "validation_rows": len(validation),
    "feature_count": len(FEATURES),
    "features": FEATURES,
    "classification_threshold": 0.50,
    "xgboost_scale_pos_weight": scale_pos_weight,
    "random_state": 42
}

import json

with open(
    REPORT_DIR / "model_training_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)

logger.info("All baseline model experiments completed.")

print("\nWeek 4 model training experiments completed.")
print("Reports saved in:", REPORT_DIR)

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
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


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

DATA_PATH = "datasets/processed/invoices_features_all.parquet"
REPORT_DIR = Path("reports/week4_modeling")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
THRESHOLD = 0.50


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)
df["posting_date"] = pd.to_datetime(df["posting_date"])

df = df[
    (df["posting_date"] <= "2019-08-31") &
    (df["late_payment"].notna())
].copy()

df["late_payment"] = df["late_payment"].astype(int)


# --------------------------------------------------
# APPROVED FEATURES
# --------------------------------------------------

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

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")

categorical = [
    "business_code",
    "invoice_currency",
    "document_type",
    "cust_payment_terms"
]

numeric = [col for col in features if col not in categorical]


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

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
            categorical
        ),
        (
            "numeric",
            Pipeline(numeric_steps),
            numeric
        )
    ])


# --------------------------------------------------
# MODEL FACTORY
# --------------------------------------------------

def create_model(name, scale_pos_weight=None):

    if name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=2
        )

    elif name in ["XGBoost unweighted", "XGBoost weighted"]:
        weight = (
            1.0 if scale_pos_weight is None
            else scale_pos_weight
        )

        model = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=weight,
            random_state=42,
            n_jobs=2
        )

    else:
        raise ValueError(f"Unknown model: {name}")

    return model


# --------------------------------------------------
# CHRONOLOGICAL CROSS-VALIDATION
# --------------------------------------------------

unique_dates = np.array(sorted(df["posting_date"].unique()))

cv = TimeSeriesSplit(n_splits=N_SPLITS)

model_names = [
    "Majority baseline",
    "Logistic Regression",
    "Random Forest",
    "XGBoost unweighted",
    "XGBoost weighted"
]

results = []

print("\nStarting five-fold expanding-window CV...")


for fold, (train_idx, val_idx) in enumerate(
    cv.split(unique_dates), start=1
):

    train_dates = unique_dates[train_idx]
    val_dates = unique_dates[val_idx]

    train_df = df[df["posting_date"].isin(train_dates)].copy()
    val_df = df[df["posting_date"].isin(val_dates)].copy()

    X_train = train_df[features]
    y_train = train_df["late_payment"]

    X_val = val_df[features]
    y_val = val_df["late_payment"]

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    scale_pos_weight = negatives / positives

    print(f"\nFOLD {fold}")
    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))

    for name in model_names:

        print("Training:", name)

        if name == "Majority baseline":

            majority_class = int(y_train.mode().iloc[0])

            predictions = np.full(
                len(y_val), majority_class
            )

            probabilities = np.full(
                len(y_val), float(majority_class)
            )

            roc_auc = np.nan
            pr_auc = np.nan

        else:

            weighted = name == "XGBoost weighted"

            model = create_model(
                name,
                scale_pos_weight=(
                    scale_pos_weight if weighted else None
                )
            )

            preprocessor = create_preprocessor(
                scale_numeric=(name == "Logistic Regression")
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            probabilities = pipeline.predict_proba(X_val)[:, 1]

            predictions = (
                probabilities >= THRESHOLD
            ).astype(int)

            roc_auc = roc_auc_score(y_val, probabilities)

            pr_auc = average_precision_score(
                y_val, probabilities
            )

        results.append({
            "fold": fold,
            "model": name,
            "training_rows": len(train_df),
            "validation_rows": len(val_df),
            "training_start": train_df["posting_date"].min().date(),
            "training_end": train_df["posting_date"].max().date(),
            "validation_start": val_df["posting_date"].min().date(),
            "validation_end": val_df["posting_date"].max().date(),
            "threshold": THRESHOLD,
            "validation_late_rate": y_val.mean(),
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
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "scale_pos_weight": (
                scale_pos_weight if name == "XGBoost weighted"
                else np.nan
            )
        })

        print(
            f"  F1: {results[-1]['f1']:.4f} | "
            f"Recall: {results[-1]['recall']:.4f}"
        )


# --------------------------------------------------
# SAVE FOLD RESULTS
# --------------------------------------------------

fold_results = pd.DataFrame(results)

fold_results.to_csv(
    REPORT_DIR / "time_cv_fold_results.csv",
    index=False
)


# --------------------------------------------------
# SUMMARY: MEAN AND STANDARD DEVIATION
# --------------------------------------------------

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc"
]

summary_rows = []

for name in model_names:

    model_data = fold_results[
        fold_results["model"] == name
    ]

    row = {
        "model": name,
        "folds": len(model_data)
    }

    for metric in metrics:
        row[f"{metric}_mean"] = model_data[metric].mean()
        row[f"{metric}_std"] = model_data[metric].std()

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    "f1_mean",
    ascending=False
)

summary.to_csv(
    REPORT_DIR / "time_cv_summary.csv",
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("FIVE-FOLD TIME-AWARE CV SUMMARY")
print("=" * 65)

print(
    summary[
        [
            "model",
            "accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "roc_auc_mean",
            "pr_auc_mean"
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print("reports/week4_modeling/time_cv_fold_results.csv")
print("reports/week4_modeling/time_cv_summary.csv")

print("\nTime-aware cross-validation completed.")

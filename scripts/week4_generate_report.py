from pathlib import Path
import pandas as pd

report_dir = Path("reports/week4_modeling")
report_dir.mkdir(parents=True, exist_ok=True)

# Results from the common September-November validation experiment
comparison = pd.DataFrame([
    {
        "model": "Majority baseline",
        "imbalance_method": "None",
        "accuracy": 0.5814,
        "precision": 0.0000,
        "recall": 0.0000,
        "f1": 0.0000,
        "roc_auc": None,
        "pr_auc": None
    },
    {
        "model": "Logistic Regression",
        "imbalance_method": "None",
        "accuracy": 0.7458,
        "precision": 0.7305,
        "recall": 0.6224,
        "f1": 0.6721,
        "roc_auc": 0.8101,
        "pr_auc": 0.7631
    },
    {
        "model": "Random Forest",
        "imbalance_method": "class_weight=balanced",
        "accuracy": 0.7942,
        "precision": 0.7577,
        "recall": 0.7473,
        "f1": 0.7525,
        "roc_auc": 0.8504,
        "pr_auc": 0.8252
    },
    {
        "model": "XGBoost",
        "imbalance_method": "Unweighted",
        "accuracy": 0.7981,
        "precision": 0.7905,
        "recall": 0.7044,
        "f1": 0.7450,
        "roc_auc": 0.8544,
        "pr_auc": 0.8350
    },
    {
        "model": "XGBoost",
        "imbalance_method": "scale_pos_weight=1.3386",
        "accuracy": 0.7968,
        "precision": 0.7716,
        "recall": 0.7308,
        "f1": 0.7506,
        "roc_auc": 0.8549,
        "pr_auc": 0.8349
    }
])

comparison.to_csv(
    report_dir / "model_comparison.csv",
    index=False
)


# Random Forest threshold experiments
thresholds = pd.DataFrame([
    [0.30, 0.6564, 0.8376, 0.7360],
    [0.35, 0.6937, 0.8142, 0.7491],
    [0.40, 0.7202, 0.7936, 0.7551],
    [0.45, 0.7334, 0.7740, 0.7532],
    [0.50, 0.7577, 0.7473, 0.7525],
    [0.55, 0.7759, 0.7165, 0.7450],
    [0.60, 0.8010, 0.6746, 0.7324],
    [0.65, 0.8373, 0.6149, 0.7091],
    [0.70, 0.8640, 0.5279, 0.6554]
], columns=["threshold", "precision", "recall", "f1"])

thresholds.to_csv(
    report_dir / "threshold_analysis.csv",
    index=False
)


# Load final test results
final_metrics = pd.read_csv(
    report_dir / "final_test_classification_metrics.csv"
)

calibration_metrics = pd.read_csv(
    report_dir / "calibration_test_metrics.csv"
)

calibrated = final_metrics[
    final_metrics["model"] ==
    "Sigmoid Calibrated Random Forest"
].iloc[0]

uncalibrated_cal = calibration_metrics.iloc[0]
calibrated_cal = calibration_metrics.iloc[1]

# Generate written summary
summary = f"""
WEEK 4 - CLASSIFICATION MODELING SUMMARY
=========================================

OBJECTIVE
Predict the probability that a settled invoice will be paid late.

MODEL COMPARISON
----------------
All initial model comparisons used the September-November 2019
validation period.

{comparison.to_string(index=False)}

MODEL SELECTION
---------------
Random Forest with balanced class weights produced the highest
validation F1 score among the tested models at the default 0.50
threshold.

XGBoost produced slightly stronger ROC-AUC and PR-AUC results.
Therefore, Random Forest is selected for the calibrated pipeline
based on its validation F1 and recall, not because it wins every
metric.

THRESHOLD ANALYSIS
------------------
The highest validation F1 among the tested Random Forest thresholds
was at 0.40:

Precision: 72.02%
Recall:    79.36%
F1:        75.51%

The 0.40 threshold is a provisional operating choice. Business costs
for false alerts and missed late payments have not been established.

PROBABILITY CALIBRATION
-----------------------
Calibration method: Sigmoid
Calibration period: December 2019-January 2020

Uncalibrated Brier score: {uncalibrated_cal['brier_score']:.6f}
Calibrated Brier score:   {calibrated_cal['brier_score']:.6f}

Uncalibrated log loss: {uncalibrated_cal['log_loss']:.6f}
Calibrated log loss:   {calibrated_cal['log_loss']:.6f}

FINAL TEST
----------
Final test period: February 1-27, 2020
Test invoices: {int(calibrated['true_negatives'] + calibrated['false_positives'] + calibrated['false_negatives'] + calibrated['true_positives'])}

Accuracy:  {calibrated['accuracy']:.4%}
Precision: {calibrated['precision']:.4%}
Recall:    {calibrated['recall']:.4%}
F1 score:  {calibrated['f1']:.4%}
ROC-AUC:   {calibrated['roc_auc']:.6f}
PR-AUC:    {calibrated['pr_auc']:.6f}

CONFUSION MATRIX
----------------
True negatives:  {int(calibrated['true_negatives'])}
False positives: {int(calibrated['false_positives'])}
False negatives: {int(calibrated['false_negatives'])}
True positives:  {int(calibrated['true_positives'])}

INTERPRETATION
--------------
The model identified approximately 77.67% of late invoices in the
final test period.

There were 353 false positive alerts and 211 missed late invoices.
Approximately 32.5% of predicted late alerts were false positives.

Calibration improved Brier score and log loss. The calibrated
classification threshold was mapped from the original 0.40
Random Forest threshold to approximately 0.34915. The final test
classification metrics remained unchanged after this mapping.

FEATURES AND DATA CONTROL
------------------------
- 17 approved model features were used.
- total_open_amount was excluded because its business meaning
  remains unconfirmed.
- Outcome fields and invoice/customer identifiers were excluded.
- Customer history features use only payment outcomes available
  before each invoice's posting date.
- Open invoices without known outcomes were not used as labelled
  training examples.

LIMITATIONS
-----------
1. Evaluation covers historical settled invoices, not all future
   open invoices. Selection bias is possible.
2. The dataset covers a limited historical period.
3. The business meaning of total_open_amount is unconfirmed.
4. The threshold has not been optimized against actual business
   costs.
5. Calibration improvement is modest and needs monitoring on new data.
6. Validation class-imbalance methods differed across models:
   Random Forest used balanced class weights, weighted XGBoost used
   scale_pos_weight, and Logistic Regression and unweighted XGBoost
   did not use class weighting.

FINAL DECISION
--------------
Use the sigmoid-calibrated Random Forest as the current Week 4
candidate model. Treat the 0.34915 calibrated threshold as a
provisional alert threshold pending business cost validation.

This is a historical evaluation, not evidence of production
performance.
"""

(report_dir / "week4_summary.txt").write_text(
    summary.strip() + "\n",
    encoding="utf-8"
)

print("Week 4 reports generated:")
print("model_comparison.csv")
print("threshold_analysis.csv")
print("week4_summary.txt")

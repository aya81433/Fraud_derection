import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, classification_report,
    confusion_matrix, average_precision_score, recall_score, f1_score, precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_fraud_dataset_ready.csv")

# Feature engineering
df['value_ratio'] = df['declared_value_usd'] / df['value_usd']
df['tax_discrepancy'] = df['tax_paid_mad'] - (df['value_usd'] * df['expected_tax_rate'])
df['weight_value_ratio'] = df['weight_kg'] / df['value_usd']

# Split features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Define column types
categorical_cols = ["product_description", "country", "importer"]
numerical_cols = ["declared_code", "weight_kg", "value_usd", "declared_value_usd", 
                 "tax_paid_mad", "expected_tax_rate", "value_ratio", "tax_discrepancy", 
                 "weight_value_ratio"]
# For leakage check, exclude potentially risky features
numerical_cols_no_leakage = ["declared_code", "weight_kg", "value_usd", "declared_value_usd", 
                            "value_ratio", "weight_value_ratio"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
     ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
    ("num", "passthrough", numerical_cols)
])
# Preprocessor for leakage check
preprocessor_no_leakage = ColumnTransformer([
     ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
    ("num", "passthrough", numerical_cols_no_leakage)
])

# Option to train on a subset for faster experimentation
use_subset = False
if use_subset:
    X, _, y, _ = train_test_split(X, y, train_size=100000, stratify=y, random_state=42)

# Split into train and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=42)

# Fit transform train set
X_train_encoded = preprocessor.fit_transform(X_train)
X_val_encoded = preprocessor.transform(X_val)
X_test_encoded = preprocessor.transform(X_test)

# Apply SMOTE + Undersampling
over = SMOTE(sampling_strategy=0.5, random_state=42)
under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
sampling_pipeline = ImbPipeline([
    ('over', over),
    ('under', under)
])
X_train_balanced, y_train_balanced = sampling_pipeline.fit_resample(X_train_encoded, y_train)

# Create calibrated LightGBM model
lgbm = LGBMClassifier(is_unbalance=True, random_state=42, n_jobs=-1)
model = CalibratedClassifierCV(estimator=lgbm, method='isotonic', cv=3)
model.fit(X_train_balanced, y_train_balanced)

# Predict probabilities
val_proba = model.predict_proba(X_val_encoded)[:, 1]
test_proba = model.predict_proba(X_test_encoded)[:, 1]

# Threshold tuning: Maximize recall, F1-score, and precision
thresholds = np.arange(0.1, 0.9, 0.05)
recall_scores = []
f1_scores = []
precision_scores = []
for t in thresholds:
    val_preds = (val_proba >= t).astype(int)
    recall_scores.append(recall_score(y_val, val_preds))
    f1_scores.append(f1_score(y_val, val_preds))
    precision_scores.append(precision_score(y_val, val_preds))

optimal_threshold_recall = thresholds[np.argmax(recall_scores)]
optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
optimal_threshold_precision = thresholds[np.argmax(precision_scores)]
print(f"Optimal threshold (max recall): {optimal_threshold_recall:.2f}")
print(f"Optimal threshold (max F1-score): {optimal_threshold_f1:.2f}")
print(f"Optimal threshold (max precision): {optimal_threshold_precision:.2f}")

# Use recall-optimized threshold (prioritize fraud detection)
optimal_threshold = optimal_threshold_recall
val_preds = (val_proba >= optimal_threshold).astype(int)
test_preds = (test_proba >= optimal_threshold).astype(int)

# Evaluation function with FPR and detailed fraud metrics
def evaluate(name, y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])  # False Positive Rate
    print(f"\n--- {name} Set Evaluation ---")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_true, y_proba):.4f}")
    print(f"PR AUC: {average_precision_score(y_true, y_proba):.4f}")
    print(f"Fraud Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"Fraud F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Fraud Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"False Positive Rate: {fpr:.6f}")

# Evaluate main model
evaluate("Validation", y_val, val_preds, val_proba)
evaluate("Test", y_test, test_preds, test_proba)

# Feature importance analysis
feature_names = preprocessor.get_feature_names_out()
importances = model.estimator_.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nFeature Importance:")
print(feature_imp_df.sort_values(by='Importance', ascending=False).head(10))

# Misclassification analysis
val_preds = (val_proba >= optimal_threshold).astype(int)
misclassified = X_val[val_preds != y_val].copy()
misclassified['is_fraud'] = y_val[val_preds != y_val]
misclassified['predicted'] = val_preds[val_preds != y_val]
print("\nMisclassification Summary (Validation Set):")
print(misclassified.groupby(['is_fraud', 'predicted'])[numerical_cols].mean())

# Data leakage check: Train model without risky features
X_train_encoded_no_leakage = preprocessor_no_leakage.fit_transform(X_train)
X_val_encoded_no_leakage = preprocessor_no_leakage.transform(X_val)
X_test_encoded_no_leakage = preprocessor_no_leakage.transform(X_test)
X_train_balanced_no_leakage, y_train_balanced_no_leakage = sampling_pipeline.fit_resample(X_train_encoded_no_leakage, y_train)
lgbm_no_leakage = LGBMClassifier(is_unbalance=True, random_state=42, n_jobs=-1)
model_no_leakage = CalibratedClassifierCV(estimator=lgbm_no_leakage, method='isotonic', cv=3)
model_no_leakage.fit(X_train_balanced_no_leakage, y_train_balanced_no_leakage)
val_proba_no_leakage = model_no_leakage.predict_proba(X_val_encoded_no_leakage)[:, 1]
test_proba_no_leakage = model_no_leakage.predict_proba(X_test_encoded_no_leakage)[:, 1]
val_preds_no_leakage = (val_proba_no_leakage >= optimal_threshold).astype(int)
test_preds_no_leakage = (test_proba_no_leakage >= optimal_threshold).astype(int)
print("\n--- Data Leakage Check (No tax_discrepancy, tax_paid_mad, expected_tax_rate) ---")
evaluate("Validation (No Leakage)", y_val, val_preds_no_leakage, val_proba_no_leakage)
evaluate("Test (No Leakage)", y_test, test_preds_no_leakage, test_proba_no_leakage)

# Check feature correlations with target
correlations = X[numerical_cols].corrwith(y)
print("\nFeature Correlations with is_fraud:")
print(correlations.sort_values(ascending=False))

# Plot Precision-Recall vs Threshold
precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f"Threshold = {optimal_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('precision_recall_curve.png')
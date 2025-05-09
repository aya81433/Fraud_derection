import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, classification_report,
    confusion_matrix, average_precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import joblib

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

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
    ("num", "passthrough", numerical_cols)
])

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

# Create calibrated XGBoost model with class weights
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", 
                    scale_pos_weight=scale_pos_weight, random_state=42)
model = CalibratedClassifierCV(estimator=xgb, method='isotonic', cv=3)
model.fit(X_train_balanced, y_train_balanced)

# Predict probabilities
val_proba = model.predict_proba(X_val_encoded)[:, 1]
test_proba = model.predict_proba(X_test_encoded)[:, 1]

# Find optimal threshold
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = []
for t in thresholds:
    val_preds = (val_proba >= t).astype(int)
    f1_scores.append(f1_score(y_val, val_preds))
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {optimal_threshold}")

# Apply threshold
val_preds = (val_proba >= optimal_threshold).astype(int)
test_preds = (test_proba >= optimal_threshold).astype(int)

# Evaluation function
def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n--- {name} Set Evaluation ---")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_true, y_proba):.4f}")
    print(f"PR AUC: {average_precision_score(y_true, y_proba):.4f}")
    print(f"Fraud Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"Fraud F1-Score: {f1_score(y_true, y_pred):.4f}")

# Evaluate
evaluate("Validation", y_val, val_preds, val_proba)
evaluate("Test", y_test, test_preds, test_proba)

# Plot Precision-Recall vs Threshold
precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f"Threshold = {optimal_threshold}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 9: Save the trained model
joblib.dump(model, "xgboost_final_model_trained.pkl")
print("✅ Final model saved as xgboost_final_model_trained.pkl")
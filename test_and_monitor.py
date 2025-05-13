import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, classification_report,
    confusion_matrix, average_precision_score, recall_score, f1_score, precision_score
)
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Validate required columns
required_cols = ['product_description', 'declared_code', 'weight_kg', 'value_usd', 'country', 
                'importer', 'declared_value_usd', 'tax_paid_mad', 'expected_tax_rate', 'is_fraud']

# Load new dataset
try:
    new_df = pd.read_csv("new_data_for_testing.csv")
except FileNotFoundError:
    print("❌ Error: new_data_for_testing.csv not found. Run create_new_data.py first.")
    exit(1)

# Validate columns
if not all(col in new_df.columns for col in required_cols):
    print(f"❌ Error: New data missing required columns. Found: {new_df.columns.tolist()}")
    exit(1)

# Data cleaning
new_df = new_df[new_df['value_usd'] > 0]  # Prevent division by zero
new_df['value_ratio'] = new_df['declared_value_usd'] / new_df['value_usd']
new_df['value_ratio'] = new_df['value_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)

# Feature engineering (same as training)
new_df['tax_discrepancy'] = new_df['tax_paid_mad'] - (new_df['value_usd'] * new_df['expected_tax_rate'])
new_df['weight_value_ratio'] = new_df['weight_kg'] / new_df['value_usd']
new_df['tax_discrepancy_ratio'] = new_df['tax_discrepancy'] / new_df['value_usd']
mean_weight_by_country = new_df.groupby('country')['weight_kg'].mean()
new_df['weight_anomaly'] = new_df.apply(lambda x: x['weight_kg'] / mean_weight_by_country[x['country']], axis=1)

# Load saved model, preprocessor, and optimal threshold
try:
    model = joblib.load("xgboost_final_model_trained2.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    optimal_threshold = joblib.load("optimal_threshold.pkl")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit(1)
print(f"Loaded model, preprocessor, and optimal threshold: {optimal_threshold:.2f}")

# Split features and target
X_new = new_df.drop("is_fraud", axis=1)
y_new = new_df["is_fraud"]

# Preprocess new data
X_new_encoded = preprocessor.transform(X_new)

# Predict probabilities and apply threshold
new_proba = model.predict_proba(X_new_encoded)[:, 1]
new_preds = (new_proba >= optimal_threshold).astype(int)

# Evaluation function
def evaluate(name, y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) > 0 else 0
    print(f"\n--- {name} Evaluation ---")
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
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'name': name,
        'accuracy': (cm[0, 0] + cm[1, 1]) / cm.sum(),
        'fraud_recall': recall_score(y_true, y_pred),
        'fraud_precision': precision_score(y_true, y_pred),
        'fraud_f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
        'fpr': fpr
    }

# Evaluate on new data
metrics = evaluate("New Data", y_new, new_preds, new_proba)

# Alerting
if metrics['fraud_recall'] < 0.95 or metrics['pr_auc'] < 0.95:
    print("⚠️ Alert: Model performance degraded! Consider retraining.")

# Save metrics
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("monitoring_metrics.csv", mode='a', header=not pd.io.common.file_exists("monitoring_metrics.csv"), index=False)
print("✅ Metrics saved to monitoring_metrics.csv")

# Feature importance
feature_names = preprocessor.get_feature_names_out()
importances = model.calibrated_classifiers_[0].estimator.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance (Top 10):")
print(feature_imp_df.head(10))

# Save feature importance
feature_imp_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
feature_imp_df.to_csv("monitoring_feature_importance.csv", mode='a', header=not pd.io.common.file_exists("monitoring_feature_importance.csv"), index=False)
print("✅ Feature importance saved to monitoring_feature_importance.csv")

# Misclassification analysis
numerical_cols = ["declared_code", "weight_kg", "value_usd", "declared_value_usd", 
                 "tax_paid_mad", "expected_tax_rate", "value_ratio", "tax_discrepancy", 
                 "weight_value_ratio", "tax_discrepancy_ratio", "weight_anomaly"]
misclassified = X_new[new_preds != y_new].copy()
misclassified['is_fraud'] = y_new[new_preds != y_new]
misclassified['predicted'] = new_preds[new_preds != y_new]
print("\nMisclassification Summary (New Data):")
misclassification_summary = misclassified.groupby(['is_fraud', 'predicted'])[numerical_cols].mean()
print(misclassification_summary)

# Save misclassification summary
misclassification_summary = misclassification_summary.reset_index()
misclassification_summary['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
misclassification_summary.to_csv("monitoring_misclassifications.csv", mode='a', header=not pd.io.common.file_exists("monitoring_misclassifications.csv"), index=False)
print("✅ Misclassification summary saved to monitoring_misclassifications.csv")

# Plot Precision-Recall vs Threshold
precision, recall, thresholds = precision_recall_curve(y_new, new_proba)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f"Threshold = {optimal_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold (New Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('precision_recall_curve_new_data.png')
print("✅ Precision-Recall curve saved as precision_recall_curve_new_data.png")

# Plot performance trends
metrics_history = pd.read_csv("monitoring_metrics.csv")
plt.figure(figsize=(10, 6))
plt.plot(metrics_history['timestamp'], metrics_history['fraud_recall'], label='Fraud Recall')
plt.plot(metrics_history['timestamp'], metrics_history['pr_auc'], label='PR AUC')
plt.xlabel("Timestamp")
plt.ylabel("Metric Value")
plt.title("Model Performance Over Time")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('performance_trend.png')
print("✅ Performance trend saved as performance_trend.png")
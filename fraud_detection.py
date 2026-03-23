import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['Class'].value_counts()}")
print(f"\nFraud percentage: {round(df['Class'].mean() * 100, 4)}%")


plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette=['steelblue', 'crimson'])
plt.title('Class Distribution (0 = Legitimate, 1 = Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
print("Saved: class_distribution.png")


plt.figure(figsize=(8, 4))
sns.boxplot(x='Class', y='Amount', data=df, palette=['steelblue', 'crimson'])
plt.title('Transaction Amount by Class')
plt.yscale('log')
plt.tight_layout()
plt.savefig('amount_by_class.png')
plt.show()
print("Saved: amount_by_class.png")


scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])


df.drop(['Amount', 'Time'], axis=1, inplace=True)


X = df.drop('Class', axis=1)
y = df['Class']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

print("\nApplying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled training class distribution:\n{pd.Series(y_train_resampled).value_counts()}")


print("\n--- Training Logistic Regression (Baseline) ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_resampled, y_train_resampled)

lr_preds = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_preds, target_names=['Legitimate', 'Fraud']))
print(f"ROC-AUC Score: {round(roc_auc_score(y_test, lr_probs), 4)}")


print("\n--- Training Random Forest Classifier ---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train_resampled, y_train_resampled)

rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds, target_names=['Legitimate', 'Fraud']))
print(f"ROC-AUC Score: {round(roc_auc_score(y_test, rf_probs), 4)}")


cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Fraud'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png')
plt.show()
print("Saved: confusion_matrix_rf.png")


lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {round(roc_auc_score(y_test, lr_probs), 2)})', color='steelblue')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {round(roc_auc_score(y_test, rf_probs), 2)})', color='crimson')
plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve_comparison.png')
plt.show()
print("Saved: roc_curve_comparison.png")


feature_importances = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='Reds_r')
plt.title('Top 15 Feature Importances - Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("Saved: feature_importance.png")


print("\n" + "="*50)
print("FINAL MODEL SUMMARY")
print("="*50)
print(f"Logistic Regression ROC-AUC : {round(roc_auc_score(y_test, lr_probs), 4)}")
print(f"Random Forest ROC-AUC       : {round(roc_auc_score(y_test, rf_probs), 4)}")
print("\nBest Model: Random Forest Classifier")
print("Outputs saved: class_distribution.png, amount_by_class.png,")
print("               confusion_matrix_rf.png, roc_curve_comparison.png,")
print("               feature_importance.png")
print("="*50)

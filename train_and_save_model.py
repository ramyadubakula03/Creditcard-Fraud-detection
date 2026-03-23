# ============================================================
# Run this ONCE to train and save the model
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

# Preprocess
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)

# Save model and scaler
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/fraud_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')

print("✅ Model saved to model/fraud_model.pkl")
print("✅ Scaler saved to model/scaler.pkl")

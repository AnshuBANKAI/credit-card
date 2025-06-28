# credit_card_fraud_detection.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 1
df = pd.read_csv("creditcard.csv")  

# Step 2
print("Dataset shape:", df.shape) 
print(df['Class'].value_counts())

# Step 3
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# Step 4
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 6
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE Resampling:")
print(pd.Series(y_train_resampled).value_counts())

# Step 7
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_resampled, y_train_resampled)
y_pred_log = log_reg.predict(X_test)

# Step 8
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_clf.predict(X_test)

# Step 9
print("\n--- Logistic Regression Classification Report ---")
print(classification_report(y_test, y_pred_log))

# Step 10
print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, y_pred_rf))

print("\n--- Confusion Matrix (Random Forest) ---")
print(confusion_matrix(y_test, y_pred_rf))

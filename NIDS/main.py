# ==============================
# NIDS.py
# Network Intrusion Detection System
# Final Year Project
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ==============================
# 1️⃣ LOAD DATASET
# ==============================
print("Loading dataset...")
df = pd.read_csv(r"dataset/nids_data.csv")
print("Dataset loaded:", df.shape)
print("\nClass distribution before cleaning:")
print(df['label'].value_counts())

# ==============================
# 2️⃣ REMOVE EXTREMELY RARE CLASSES (<20 samples)
# ==============================
print("\nRemoving rare classes (<20 samples)...")gt
class_counts = df['label'].value_counts()
valid_classes = class_counts[class_counts >= 20].index
df = df[df['label'].isin(valid_classes)]


print("\nClass distribution after cleaning:")
print(df['label'].value_counts())

# ==============================
# 3️⃣ SPLIT FEATURES & TARGET
# ==============================
X = df.drop('label', axis=1)
y = df['label']

# ==============================
# 4️⃣ ENCODE CATEGORICAL FEATURES
# ==============================
categorical_cols = ['protocol_type', 'service', 'flag']
X = pd.get_dummies(X, columns=categorical_cols)

print("\nFeature shape after encoding:", X.shape)

# ==============================
# 5️⃣ TRAIN-TEST SPLIT (STRATIFIED)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)

# ==============================
# 6️⃣ FEATURE SCALING (For LR & SVM)
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 7️⃣ RANDOM FOREST
# ==============================
print("\n===== Random Forest =====")
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ==============================
# 8️⃣ LOGISTIC REGRESSION
# ==============================
print("\n===== Logistic Regression =====")
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# ==============================
# 9️⃣ SUPPORT VECTOR MACHINE
# ==============================
print("\n===== Support Vector Machine =====")
svm = SVC(
    kernel='rbf',
    class_weight='balanced'
)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# ==============================
# 🔟 CONFUSION MATRIX (Random Forest)
# ==============================
print("\nGenerating Confusion Matrix for Random Forest...")
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 1️⃣1️⃣ FEATURE IMPORTANCE (Random Forest)
# ==============================
print("\nGenerating Feature Importance Plot...")
importances = rf.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names)
top_features = feat_imp.nlargest(10)

plt.figure(figsize=(8, 6))
top_features.sort_values().plot(kind='barh')
plt.title("Top 10 Important Features (Random Forest)")
plt.tight_layout()
plt.show()

print("\nNIDS Model Training Completed Successfully!")
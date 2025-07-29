import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.utils import resample
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

#Load Dataset 

data_folder = './data'
print("Loading data files...")
all_data = []

for file in tqdm(sorted(os.listdir(data_folder))):
    if file.endswith('.pkl'):
        df = pd.read_pickle(os.path.join(data_folder, file))
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
print(f"Loaded {len(data):,} transactions from {len(all_data)} files.")

# Convert datetime and sort
data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'])
data.sort_values(by='TX_DATETIME', inplace=True)
data.reset_index(drop=True, inplace=True)

# Feature Engineering 

# Feature 1: Amount greater than 220
data['AMOUNT_OVER_220'] = (data['TX_AMOUNT'] > 220).astype(int)

# Feature 2: Terminal fraud count in past 28 days
print("Calculating terminal fraud history...")
data['TERMINAL_FRAUD_COUNT'] = 0
terminal_fraud = {}

for idx, row in tqdm(data.iterrows(), total=len(data)):
    term_id = row['TERMINAL_ID']
    tx_date = row['TX_DATETIME']

    if term_id not in terminal_fraud:
        terminal_fraud[term_id] = []

    terminal_fraud[term_id] = [d for d in terminal_fraud[term_id] if (tx_date - d).days <= 28]

    if row['TX_FRAUD'] == 1:
        terminal_fraud[term_id].append(tx_date)

    data.at[idx, 'TERMINAL_FRAUD_COUNT'] = len(terminal_fraud[term_id])

# Feature 3: Spending ratio
print("Calculating customer spending behavior...")
data['CUSTOMER_MEAN_SPEND'] = data.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform('mean')
data['SPENDING_RATIO'] = data['TX_AMOUNT'] / (data['CUSTOMER_MEAN_SPEND'] + 1e-3)

# Define Features 

features = ['TX_AMOUNT', 'AMOUNT_OVER_220', 'TERMINAL_FRAUD_COUNT', 'SPENDING_RATIO']
target = 'TX_FRAUD'

# Train-Test Split 

cutoff_date = data['TX_DATETIME'].quantile(0.8)
train_data = data[data['TX_DATETIME'] <= cutoff_date]
test_data = data[data['TX_DATETIME'] > cutoff_date]

# Show class distribution
print("\nClass distribution in training set:")
print(train_data['TX_FRAUD'].value_counts())

print("\nClass distribution in test set:")
print(test_data['TX_FRAUD'].value_counts())

# Downsample Legit Transactions 

print("\nBalancing the training dataset...")
legit = train_data[train_data['TX_FRAUD'] == 0]
fraud = train_data[train_data['TX_FRAUD'] == 1]

legit_downsampled = resample(
    legit,
    replace=False,
    n_samples=len(fraud) * 5,  
    random_state=42
)

train_balanced = pd.concat([fraud, legit_downsampled])
train_balanced = train_balanced.sample(frac=1, random_state=42) 

X_train = train_balanced[features]
y_train = train_balanced[target]
X_test = test_data[features]
y_test = test_data[target]

print(f"\nTraining on {len(X_train):,} rows, testing on {len(X_test):,} rows.")

# Train Model

print("\nTraining Random Forest model...")
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Evaluate Model 

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Feature Importance 

feature_importance = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)

feature_importance.plot(kind='barh', title='Feature Importance')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Save Model 

model_path = 'fraud_detection_model.pkl'
joblib.dump(clf, model_path)
print(f"\nModel saved to: {model_path}")

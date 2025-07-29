import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
from datetime import timedelta

# Config
MODEL_PATH = 'fraud_detection_model.pkl'
DATA_FOLDER = 'data'
THRESHOLD = 0.4
OUTPUT_CSV = 'fraud_summary_report.csv'

print(f"📦 Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

summary = []

# Process each file
for filename in tqdm(sorted(os.listdir(DATA_FOLDER))):
    if filename.endswith('.pkl'):
        filepath = os.path.join(DATA_FOLDER, filename)
        df = pd.read_pickle(filepath)

        if df.empty:
            continue

        # Feature Engineering
        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        df['AMOUNT_OVER_220'] = (df['TX_AMOUNT'] > 220).astype(int)

        # Compute TERMINAL_FRAUD_COUNT
        df['TERMINAL_FRAUD_COUNT'] = 0
        terminal_fraud = {}

        for idx, row in df.iterrows():
            term_id = row['TERMINAL_ID']
            tx_date = row['TX_DATETIME']

            if term_id not in terminal_fraud:
                terminal_fraud[term_id] = []

            # Remove old frauds
            terminal_fraud[term_id] = [d for d in terminal_fraud[term_id] if (tx_date - d).days <= 28]

            if 'TX_FRAUD' in df.columns and row['TX_FRAUD'] == 1:
                terminal_fraud[term_id].append(tx_date)

            df.at[idx, 'TERMINAL_FRAUD_COUNT'] = len(terminal_fraud[term_id])

        # Spending ratio
        df['CUSTOMER_MEAN_SPEND'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform('mean')
        df['SPENDING_RATIO'] = df['TX_AMOUNT'] / (df['CUSTOMER_MEAN_SPEND'] + 1e-3)

        features = ['TX_AMOUNT', 'AMOUNT_OVER_220', 'TERMINAL_FRAUD_COUNT', 'SPENDING_RATIO']
        X = df[features]

        # Predict 
        df['FRAUD_PROB'] = model.predict_proba(X)[:, 1]
        df['IS_FRAUD'] = (df['FRAUD_PROB'] >= THRESHOLD).astype(int)

        # Evaluation 
        if 'TX_FRAUD' in df.columns:
            y_true = df['TX_FRAUD']
            y_pred = df['IS_FRAUD']
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            actual_frauds = y_true.sum()
            predicted_frauds = y_pred.sum()
            correct_predictions = ((y_true == 1) & (y_pred == 1)).sum()
        else:
            precision = recall = f1 = np.nan
            actual_frauds = predicted_frauds = correct_predictions = np.nan

        summary.append({
            'File': filename,
            'Total_Transactions': len(df),
            'Actual_Frauds': actual_frauds,
            'Predicted_Frauds': predicted_frauds,
            'Correct_Predictions': correct_predictions,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        })

# Save Summary Report
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUTPUT_CSV, index=False)

# Final Stats
total_tx = summary_df['Total_Transactions'].sum()
total_actual = summary_df['Actual_Frauds'].sum()
total_predicted = summary_df['Predicted_Frauds'].sum()
total_correct = summary_df['Correct_Predictions'].sum()
avg_precision = summary_df['Precision'].mean()
avg_recall = summary_df['Recall'].mean()
avg_f1 = summary_df['F1_Score'].mean()

print("\n FINAL SUMMARY:")
print(f" Files processed         : {len(summary_df)}")
print(f" Total Transactions      : {total_tx:,}")
print(f" Total Actual Frauds     : {total_actual:,}")
print(f" Total Predicted Frauds  : {total_predicted:,}")
print(f" Correct Fraud Predictions: {total_correct:,}")
print(f" Avg Precision           : {avg_precision:.4f}")
print(f" Avg Recall              : {avg_recall:.4f}")
print(f" Avg F1-Score            : {avg_f1:.4f}")

print(f"\n CSV summary saved to: {OUTPUT_CSV}")

#  Fraud Detection in Financial Transactions using Machine Learning

This project implements a fraud detection system using a RandomForest classifier on anonymized financial transaction data stored in `.pkl` files. The system predicts the likelihood of fraud for each transaction and summarizes results across multiple days.

---

##  Project Goals

- Detect suspicious or fraudulent transactions.
- Learn practical data processing from raw `.pkl` files.
- Handle real-world challenges like class imbalance and feature engineering.
- Evaluate model performance and summarize results across a dataset.

---

##  Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- Joblib (model persistence)
- TQDM (progress bar)
- Jupyter/VS Code for development

---

##  How the System Works

### 1. `fraud_detection.py` – Model Training

- Loads all 183 `.pkl` transaction files from `/data`.
- Combines them and sorts by transaction time.
- Performs **feature engineering**:
  - `AMOUNT_OVER_220`: Flag for transactions over ₹220
  - `TERMINAL_FRAUD_COUNT`: Number of frauds in past 28 days at terminal
  - `SPENDING_RATIO`: TX_AMOUNT / avg customer spend
- Splits data by time: 80% for training, 20% for testing.
- Handles class imbalance using `class_weight='balanced'`.
- Trains a `RandomForestClassifier` and evaluates using:
  - Confusion Matrix
  - Classification Report
  - ROC AUC Score
- Saves model as `fraud_detection_model.pkl`.

### 2. `run_model.py` – Single Day Prediction

- Loads one `.pkl` file (e.g., `2018-08-02.pkl`).
- Performs **same feature engineering logic**.
- Loads the saved model and predicts fraud probability.
- Adds:
  - `FRAUD_PROB`: model's probability
  - `IS_FRAUD`: 1 if `FRAUD_PROB > 0.5`, else 0
- If labels (`TX_FRAUD`) are available:
  - Shows evaluation metrics.
- Useful for testing predictions on a specific day.

### 3. `batch_predict_summary.py` – All Files, Summary Report

- Iterates over all 183 `.pkl` files.
- Runs the prediction pipeline with threshold tuning (`FRAUD_PROB > 0.4`).
- Aggregates precision, recall, F1-score, fraud counts per file.
- Saves `fraud_summary_report.csv` containing:
  - Filename
  - Total transactions
  - Actual and predicted frauds
  - Correct predictions
  - Evaluation scores

---

###  Sample Output (Terminal Summary)

- FINAL SUMMARY:
- Files processed : 183  
- Total Transactions : 1,754,155  
- Total Actual Frauds : 11,200  
- Total Predicted Frauds : 9,482  
- Correct Fraud Predictions: 8,523  
- Avg Precision : 0.82  
- Avg Recall : 0.76  
- Avg F1-Score : 0.78  
- CSV summary saved to: `fraud_summary_report.csv`

---

##  Challenges Faced

- **High Class Imbalance**: Only ~0.5% of transactions are fraud, which required `class_weight='balanced'` and sampling strategies.
- **Time-based Features**: `TERMINAL_FRAUD_COUNT` required accurate chronological history per terminal.
- **Memory Efficiency**: With 1.7M rows, printing too much or storing intermediate predictions was heavy. Needed to batch process files and reduce console clutter.
- **Data Consistency**: Ensuring feature engineering during prediction exactly matched the training phase logic.
- **Threshold Tuning**: Found that using a lower threshold (e.g., 0.4) improved recall significantly without sacrificing too much precision.

---

##  What I Learned

- Implementing **end-to-end ML pipeline**: data ingestion → feature engineering → model training → inference → evaluation.
- Deep understanding of **imbalanced classification**.
- Importance of **robust feature engineering** especially in fraud and anomaly detection domains.
- Experience working with **real-world formats** (e.g., `.pkl`) and handling datasets over time.
- Automated report generation and performance monitoring across time-series structured data.

---

##  Requirements

```bash
pip install -r requirements.txt
```
---
###  How to run
## step 1: Clone the repo

```bash
git clone https://github.com/your-username/fraud-detection-project.git
cd fraud-detection-project
```
## Step 2: Create and Activate Virtual Environment

```bash
python -m venv venv
# Activate it:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```
## Step 3: Install Requirements

```bash
pip install -r requirements.txt

```
## Step 4: Train the Model (Optional – if model not already trained)

```bash
python fraud_detection.py
```

## Step 5: Run the Script

```bash
python run_model.py
```

---
### This will generate a report:
📄 fraud_summary_report.csv
containing fraud predictions, metrics, and summary for each file.

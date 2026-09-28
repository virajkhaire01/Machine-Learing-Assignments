"""
Assignment 3: Support Vector Machine (SVM) for Email Classification
Model: Support Vector Classifier (SVC with RBF Kernel)
Dataset: UCI Spambase (UCI ID: 94)
Target: Class (1 = Spam, 0 = Not Spam)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Set reproducible random seed
RANDOM_STATE = 42

def setup_paths():
    """Resolve script and dataset paths relative to repository root."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dataset_path = os.path.join(script_dir, "..", "datasets", "assignment_3", "spambase.csv")
    return script_dir, os.path.abspath(local_dataset_path)

def load_data(dataset_path):
    """Load dataset from local CSV or fetch from UCI ML repository."""
    if os.path.exists(dataset_path):
        print(f"[INFO] Loading dataset from local cache: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("[INFO] Local file not found. Fetching from UCI ML Repository (ID: 94)...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=94)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        df.to_csv(dataset_path, index=False)
        print(f"[INFO] Saved dataset to {dataset_path}")
    return df

def run_assignment():
    script_dir, dataset_path = setup_paths()
    log_file = os.path.join(script_dir, "execution_log.txt")
    
    class TeeLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    sys.stdout = TeeLogger(log_file)
    
    print("=" * 75)
    print("ASSIGNMENT 3: SUPPORT VECTOR MACHINE (SVM) FOR EMAIL CLASSIFICATION")
    print("Dataset: UCI Spambase (ID: 94)")
    print("Target: Email Class (1 = Spam, 0 = Not Spam)")
    print("=" * 75)
    
    # 1. Load Data
    df = load_data(dataset_path)
    print(f"\n[1] Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Determine target column
    target_col = 'Class' if 'Class' in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    
    print(f"    Features count: {len(feature_cols)}")
    print(f"    Target column:  '{target_col}'")
    
    class_counts = df[target_col].value_counts().sort_index()
    print("\nClass Distribution:")
    print(f"    Class 0 (Not Spam / Ham): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.2f}%)")
    print(f"    Class 1 (Spam):           {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.2f}%)")
    
    # 2. Missing Value Check
    missing_count = df.isnull().sum().sum()
    print(f"\n[2] Missing Values in Dataset: {missing_count}")
    if missing_count > 0:
        df = df.dropna().copy()
        print(f"    Cleaned rows after dropna: {len(df)}")
        
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 3. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[3] Stratified Train/Test Split (80/20):")
    print(f"    Training samples: {len(X_train)} (Spam: {sum(y_train == 1)}, Ham: {sum(y_train == 0)})")
    print(f"    Testing samples:  {len(X_test)} (Spam: {sum(y_test == 1)}, Ham: {sum(y_test == 0)})")
    
    # 4. Feature Scaling using StandardScaler
    print("\n[4] Performing Feature Scaling via StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("    Scaler fitted on training data and applied to both train & test sets.")
    
    # 5. Model Training (SVM Classifier)
    print("\n[5] Training Support Vector Classifier (SVC)...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)
    svm_model.fit(X_train_scaled, y_train)
    print(f"    Fitted SVC Parameters: kernel={svm_model.kernel}, C={svm_model.C}, gamma={svm_model.gamma}")
    
    # 6. Predictions & Evaluation
    y_pred = svm_model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    prec_macro = precision_score(y_test, y_pred, average='macro')
    rec_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print("\n[6] Evaluation Metrics on Unseen Test Set:")
    print(f"    Accuracy:                 {acc:.4f} ({acc*100:.2f}%)")
    print(f"    Precision (Spam Class 1): {prec:.4f} ({prec*100:.2f}%)")
    print(f"    Recall (Spam Class 1):    {rec:.4f} ({rec*100:.2f}%)")
    print(f"    F1-Score (Spam Class 1):  {f1:.4f} ({f1*100:.2f}%)")
    print(f"    Macro F1-Score:           {f1_macro:.4f}")
    
    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix:")
    print(f"    True Negative (Non-Spam correctly identified):  {tn}")
    print(f"    False Positive (Non-Spam incorrectly marked):   {fp}")
    print(f"    False Negative (Spam missed by filter):         {fn}")
    print(f"    True Positive (Spam correctly blocked):         {tp}")
    
    # 8. Classification Report
    cls_report = classification_report(y_test, y_pred, target_names=["Not Spam (0)", "Spam (1)"], digits=4)
    print("\nDetailed Classification Report:")
    print(cls_report)
    
    report_path = os.path.join(script_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT - SVM (UCI SPAMBASE)\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_report)
    print(f"[7] Saved classification report to: {report_path}")
    
    # 9. Plot and Save Confusion Matrix
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=["Not Spam (Ham)", "Spam"],
        yticklabels=["Not Spam (Ham)", "Spam"],
        annot_kws={"size": 14, "fontweight": "bold"}
    )
    plt.title("SVM Confusion Matrix - UCI Spambase", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    
    cm_path = os.path.join(script_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[8] Saved confusion matrix visualization to: {cm_path}")
    
    # 10. Save Metrics to CSV
    metrics_path = os.path.join(script_dir, "metrics.csv")
    metrics_df = pd.DataFrame([
        {"Metric": "Accuracy", "Value": f"{acc:.4f}"},
        {"Metric": "Precision_Spam", "Value": f"{prec:.4f}"},
        {"Metric": "Recall_Spam", "Value": f"{rec:.4f}"},
        {"Metric": "F1_Score_Spam", "Value": f"{f1:.4f}"},
        {"Metric": "Precision_Macro", "Value": f"{prec_macro:.4f}"},
        {"Metric": "Recall_Macro", "Value": f"{rec_macro:.4f}"},
        {"Metric": "F1_Score_Macro", "Value": f"{f1_macro:.4f}"},
        {"Metric": "True_Negatives", "Value": str(tn)},
        {"Metric": "False_Positives", "Value": str(fp)},
        {"Metric": "False_Negatives", "Value": str(fn)},
        {"Metric": "True_Positives", "Value": str(tp)}
    ])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[9] Saved metrics table to: {metrics_path}")
    
    # 11. Save Predictions to CSV
    pred_path = os.path.join(script_dir, "predictions.csv")
    pred_df = pd.DataFrame({
        "Actual_Class": y_test,
        "Predicted_Class": y_pred,
        "Actual_Label": ["Spam" if x == 1 else "Not Spam" for x in y_test],
        "Predicted_Label": ["Spam" if x == 1 else "Not Spam" for x in y_pred],
        "Is_Correct": y_test == y_pred
    })
    pred_df.to_csv(pred_path, index=False)
    print(f"[10] Saved predictions to: {pred_path}")
    
    # 12. Write dataset_info.txt
    dataset_info_path = os.path.join(script_dir, "dataset_info.txt")
    dataset_info_content = f"""======================================================================
UCI MACHINE LEARNING REPOSITORY DATASET SPECIFICATION
======================================================================
Assignment:        Assignment 3 - SVM for Email Classification
Dataset Name:      Spambase
UCI Dataset ID:    94
UCI URL:           https://archive.ics.uci.edu/dataset/94/spambase
Donor:             Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt (HP Labs)
Total Instances:   {len(df)} emails
Input Features:    57 continuous attributes
Target Class:      Class (1 = Spam, 0 = Not Spam / Ham)

FEATURE BREAKDOWN:
----------------------------------------------------------------------
- 48 continuous features: Percentage of words in the email that match a specific word
  (e.g., word_freq_make, word_freq_address, word_freq_free, word_freq_money, word_freq_credit).
- 6 continuous features: Percentage of characters in the email that match specific characters
  (e.g., char_freq_semicolon, char_freq_dollar, char_freq_bang).
- 3 continuous features: Capital run-length metrics
  - capital_run_length_average: Average length of uninterrupted sequences of capital letters.
  - capital_run_length_longest: Maximum length of uninterrupted sequences of capital letters.
  - capital_run_length_total: Total number of capital letters in the email.

PREPROCESSING APPLIED:
----------------------------------------------------------------------
- Missing values: Checked all 58 columns; 0 missing values detected.
- Stratified Split: 80% train ({len(X_train)} samples), 20% test ({len(X_test)} samples), preserving spam ratio.
- Feature Scaling: Standardized using StandardScaler (zero mean, unit variance) to prevent high-range
  capital run-length features from overpowering percentage-frequency features.
- Random Seed: Fixed at {RANDOM_STATE} for total reproducibility.
======================================================================
"""
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        f.write(dataset_info_content)
    print(f"[11] Saved dataset audit documentation to: {dataset_info_path}")
    
    # 13. Write results.txt
    results_path = os.path.join(script_dir, "results.txt")
    results_content = f"""======================================================================
ASSIGNMENT 3: SVM FOR EMAIL CLASSIFICATION RESULTS REPORT
======================================================================
Assignment Title:      Support Vector Machine for Email Classification
Dataset Used:          UCI Spambase (Dataset ID: 94)
Algorithm:             Support Vector Classifier (SVC)
Kernel:                Radial Basis Function (RBF)
Regularization (C):    1.0
Kernel Gamma:          scale
Train/Test Split:      80% Train ({len(X_train)} samples), 20% Test ({len(X_test)} samples)
Stratification:        Enabled (preserved ~39.4% Spam ratio)
Preprocessing:         StandardScaler (Z-score normalization)
Random State:          {RANDOM_STATE}

FINAL EVALUATION METRICS (TEST SET):
----------------------------------------------------------------------
Accuracy:              {acc*100:.2f}% ({acc:.4f})
Precision (Spam):      {prec*100:.2f}% ({prec:.4f})
Recall (Spam):         {rec*100:.2f}% ({rec:.4f})
F1 Score (Spam):       {f1*100:.2f}% ({f1:.4f})
Macro F1 Score:        {f1_macro*100:.2f}% ({f1_macro:.4f})

CONFUSION MATRIX SUMMARY:
----------------------------------------------------------------------
True Negatives (Legitimate Ham Correctly Allowed): {tn}
False Positives (Legitimate Ham Flagged as Spam): {fp}
False Negatives (Spam Leaked to Inbox):            {fn}
True Positives (Spam Correctly Filtered):          {tp}

SHORT INTERPRETATION OF RESULT:
----------------------------------------------------------------------
1. Predictive Reliability: The RBF-kernel SVM achieved an outstanding classification
   accuracy of {acc*100:.2f}% on unseen emails, confirming the effectiveness of margin
   maximization in high-dimensional word/character frequency feature spaces (57 dimensions).
2. Spam Precision vs. Recall: The model obtained a precision of {prec*100:.2f}% and a
   recall of {rec*100:.2f}% on spam detection. In email security, high precision is vital
   to prevent legitimate correspondence (ham) from being erroneously quarantined; our model
   demonstrated only {fp} false positive errors out of {tn+fp} legitimate emails.
3. Importance of Scaling: StandardScaler was crucial because capital run-length attributes
   span values up to thousands, whereas word frequency percentages range from 0 to 100.
======================================================================
"""
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(results_content)
    print(f"[12] Saved final results summary to: {results_path}")
    
    print("\n" + "=" * 75)
    print("ASSIGNMENT 3 EXECUTION COMPLETE - ALL OUTPUTS GENERATED")
    print("=" * 75)

if __name__ == "__main__":
    run_assignment()

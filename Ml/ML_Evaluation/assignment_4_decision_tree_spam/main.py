"""
Assignment 4: Decision Tree for Spam / Not Spam Classification
Model: DecisionTreeClassifier
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
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

def setup_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dataset_path = os.path.join(script_dir, "..", "datasets", "assignment_4", "spambase.csv")
    return script_dir, os.path.abspath(local_dataset_path)

def load_data(dataset_path):
    if os.path.exists(dataset_path):
        print(f"[INFO] Loading dataset from local cache: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("[INFO] Fetching from UCI ML Repository (ID: 94)...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=94)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        df.to_csv(dataset_path, index=False)
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
    print("ASSIGNMENT 4: DECISION TREE FOR SPAM / NOT SPAM CLASSIFICATION")
    print("Dataset: UCI Spambase (ID: 94)")
    print("=" * 75)

    # 1. Load Data
    df = load_data(dataset_path)
    print(f"\n[1] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    target_col = 'Class' if 'Class' in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    # 2. Check missing values
    missing = df.isnull().sum().sum()
    print(f"[2] Missing Values: {missing}")
    if missing > 0:
        df = df.dropna()

    X = df[feature_cols].values
    y = df[target_col].values

    print(f"    Class 0 (Ham): {(y==0).sum()} | Class 1 (Spam): {(y==1).sum()}")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[3] Train/Test Split: {len(X_train)} train | {len(X_test)} test")

    # 4. Train Decision Tree (max_depth=8 for readable visualization)
    MAX_DEPTH = 8
    print(f"\n[4] Training DecisionTreeClassifier (max_depth={MAX_DEPTH})...")
    dt_model = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        criterion='gini',
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    dt_model.fit(X_train, y_train)
    print(f"    Tree trained. Nodes: {dt_model.tree_.node_count}, Leaves: {dt_model.get_n_leaves()}")
    print(f"    Actual depth used: {dt_model.get_depth()}")

    # 5. Prediction & Evaluation
    y_pred = dt_model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec  = recall_score(y_test, y_pred, average='binary')
    f1   = f1_score(y_test, y_pred, average='binary')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"\n[5] Evaluation Metrics:")
    print(f"    Accuracy:           {acc*100:.2f}% ({acc:.4f})")
    print(f"    Precision (Spam):   {prec*100:.2f}% ({prec:.4f})")
    print(f"    Recall (Spam):      {rec*100:.2f}% ({rec:.4f})")
    print(f"    F1-Score (Spam):    {f1*100:.2f}% ({f1:.4f})")
    print(f"    Macro F1:           {f1_macro:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    cls_report = classification_report(y_test, y_pred,
                                       target_names=["Not Spam (0)", "Spam (1)"], digits=4)
    print("\nClassification Report:")
    print(cls_report)

    # 6. Save classification report
    report_path = os.path.join(script_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT - DECISION TREE (UCI SPAMBASE)\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_report)
    print(f"[6] Saved classification report: {report_path}")

    # 7. Confusion Matrix Plot
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar=False,
                xticklabels=["Ham (0)", "Spam (1)"],
                yticklabels=["Ham (0)", "Spam (1)"],
                annot_kws={"size": 14, "fontweight": "bold"})
    plt.title("Decision Tree Confusion Matrix — UCI Spambase", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    cm_path = os.path.join(script_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[7] Saved confusion matrix: {cm_path}")

    # 8. Decision Tree Visualization (limited depth for readability)
    PLOT_DEPTH = 4
    print(f"\n[8] Generating Decision Tree visualization (plot depth={PLOT_DEPTH})...")
    short_names = [c.replace("word_freq_", "wf_").replace("char_freq_", "cf_")
                   .replace("capital_run_length_", "crl_") for c in feature_cols]

    fig, ax = plt.subplots(figsize=(26, 12), dpi=200)
    plot_tree(
        dt_model,
        feature_names=short_names,
        class_names=["Ham", "Spam"],
        filled=True,
        rounded=True,
        max_depth=PLOT_DEPTH,
        fontsize=7,
        ax=ax,
        impurity=True,
        proportion=False
    )
    plt.title(
        f"Decision Tree for Spam Classification (UCI Spambase)\n"
        f"Showing top {PLOT_DEPTH} levels of {dt_model.get_depth()} total levels | Max configured depth={MAX_DEPTH}",
        fontsize=13, fontweight="bold", pad=10
    )
    tree_path = os.path.join(script_dir, "tree.png")
    plt.savefig(tree_path, bbox_inches="tight")
    plt.close()
    print(f"    Saved tree visualization: {tree_path}")

    # 9. Feature Importance (Top 20)
    importances = dt_model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": short_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(20)

    print("\nTop 20 Most Important Features:")
    print(fi_df.to_string(index=False))

    plt.figure(figsize=(10, 7), dpi=300)
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis")
    plt.title("Decision Tree Feature Importances (Top 20) — Spambase", fontsize=13,
              fontweight="bold", pad=10)
    plt.xlabel("Gini Importance Score", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    fi_path = os.path.join(script_dir, "feature_importance.png")
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    print(f"    Saved feature importance chart: {fi_path}")

    # 10. Save metrics CSV
    metrics_path = os.path.join(script_dir, "metrics.csv")
    pd.DataFrame([
        {"Metric": "Accuracy", "Value": f"{acc:.4f}"},
        {"Metric": "Precision_Spam", "Value": f"{prec:.4f}"},
        {"Metric": "Recall_Spam", "Value": f"{rec:.4f}"},
        {"Metric": "F1_Spam", "Value": f"{f1:.4f}"},
        {"Metric": "Macro_F1", "Value": f"{f1_macro:.4f}"},
        {"Metric": "True_Negatives", "Value": str(tn)},
        {"Metric": "False_Positives", "Value": str(fp)},
        {"Metric": "False_Negatives", "Value": str(fn)},
        {"Metric": "True_Positives", "Value": str(tp)},
        {"Metric": "Max_Depth_Param", "Value": str(MAX_DEPTH)},
        {"Metric": "Actual_Depth", "Value": str(dt_model.get_depth())},
        {"Metric": "Num_Leaves", "Value": str(dt_model.get_n_leaves())}
    ]).to_csv(metrics_path, index=False)
    print(f"\n[9] Saved metrics CSV: {metrics_path}")

    # 11. Write dataset_info.txt
    dataset_info_path = os.path.join(script_dir, "dataset_info.txt")
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        f.write(f"""======================================================================
UCI MACHINE LEARNING REPOSITORY DATASET SPECIFICATION
======================================================================
Assignment:        Assignment 4 - Decision Tree for Spam Classification
Dataset Name:      Spambase
UCI Dataset ID:    94
UCI URL:           https://archive.ics.uci.edu/dataset/94/spambase
Total Instances:   {len(df)} emails
Input Features:    57 continuous word/char/capital-run-length attributes
Target Class:      Class (1 = Spam, 0 = Ham)

MODEL PARAMETERS:
----------------------------------------------------------------------
Algorithm:         DecisionTreeClassifier (scikit-learn)
Criterion:         Gini Impurity
max_depth:         {MAX_DEPTH}  (to keep visualization readable)
min_samples_split: 5
min_samples_leaf:  2
Random State:      {RANDOM_STATE}
Actual Depth:      {dt_model.get_depth()}
Number of Leaves:  {dt_model.get_n_leaves()}

PREPROCESSING:
----------------------------------------------------------------------
- Missing values: {missing} found; no imputation required.
- Stratified 80/20 train/test split.
- No feature scaling applied (Decision Trees are scale-invariant).
======================================================================
""")
    print(f"[10] Saved dataset info: {dataset_info_path}")

    # 12. Write results.txt
    results_path = os.path.join(script_dir, "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"""======================================================================
ASSIGNMENT 4: DECISION TREE SPAM CLASSIFICATION RESULTS REPORT
======================================================================
Algorithm:         DecisionTreeClassifier (Gini, max_depth={MAX_DEPTH})
Dataset:           UCI Spambase (ID: 94)
Train/Test Split:  80% ({len(X_train)}) / 20% ({len(X_test)})
Random State:      {RANDOM_STATE}

FINAL METRICS (TEST SET):
----------------------------------------------------------------------
Accuracy:          {acc*100:.2f}%
Precision (Spam):  {prec*100:.2f}%
Recall (Spam):     {rec*100:.2f}%
F1 Score (Spam):   {f1*100:.2f}%
Macro F1:          {f1_macro*100:.2f}%

Confusion Matrix:
  TN={tn}  FP={fp}
  FN={fn}  TP={tp}

INTERPRETATION:
----------------------------------------------------------------------
The Decision Tree classifier (max_depth={MAX_DEPTH}) achieves {acc*100:.2f}% accuracy on
the 921-email unseen test set. The tree's top split feature is
'{fi_df.iloc[0]["Feature"]}' (importance: {fi_df.iloc[0]["Importance"]:.4f}),
consistent with the known high discriminative value of dollar-sign
frequency and certain word frequencies for spam detection.
max_depth was set to {MAX_DEPTH} to prevent overfitting; the full-depth
tree on the training set would memorise the data. Tree visualisation
is rendered at depth {PLOT_DEPTH} for readability.
======================================================================
""")
    print(f"[11] Saved results: {results_path}")

    print("\n" + "=" * 75)
    print("ASSIGNMENT 4 EXECUTION COMPLETE - ALL OUTPUTS GENERATED")
    print("=" * 75)

if __name__ == "__main__":
    run_assignment()

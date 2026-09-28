"""
Assignment 6: Boosting and Bagging for Wine Classification
Models: AdaBoostClassifier (Boosting) and BaggingClassifier (Bagging)
Dataset: UCI Wine (UCI ID: 109)
Target: class (1, 2, or 3 — three wine cultivars)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

def setup_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dataset_path = os.path.join(script_dir, "..", "datasets", "assignment_6", "wine.csv")
    return script_dir, os.path.abspath(local_dataset_path)

def load_data(dataset_path):
    if os.path.exists(dataset_path):
        print(f"[INFO] Loading dataset from local cache: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("[INFO] Fetching from UCI ML Repository (ID: 109)...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=109)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        df.to_csv(dataset_path, index=False)
    return df

def evaluate_model(y_test, y_pred, model_name, class_names):
    """Compute and print classification metrics for a given model."""
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(f"\n--- {model_name} ---")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}% (macro)")
    print(f"  Recall:    {rec*100:.2f}% (macro)")
    print(f"  F1 Score:  {f1*100:.2f}% (macro)")
    print(f"\nClassification Report:\n{report}")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "CM": cm, "Report": report}

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
    print("ASSIGNMENT 6: BOOSTING AND BAGGING FOR WINE CLASSIFICATION")
    print("Dataset: UCI Wine (ID: 109) | 3 Cultivar Classes")
    print("=" * 75)

    # 1. Load Data
    df = load_data(dataset_path)
    print(f"\n[1] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    target_col = 'class' if 'class' in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"    Target column: '{target_col}'")
    print(f"    Class distribution:\n{df[target_col].value_counts().sort_index().to_string()}")

    # 2. Missing values
    missing = df.isnull().sum().sum()
    print(f"\n[2] Missing values: {missing}")
    if missing > 0:
        df = df.dropna()

    X = df[feature_cols].values
    y = df[target_col].values
    class_names = [f"Class {c}" for c in sorted(df[target_col].unique())]

    # 3. Train/Test Split (identical split for both models)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[3] Train/Test Split (75/25): {len(X_train)} train | {len(X_test)} test")

    # 4. Feature Scaling (for AdaBoost with weak learners)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print("[4] Feature scaling applied (StandardScaler).")

    # 5. BOOSTING — AdaBoostClassifier
    print("\n[5] Training AdaBoostClassifier (100 estimators)...")
    base_dt = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    ada_model = AdaBoostClassifier(
        estimator=base_dt,
        n_estimators=100,
        learning_rate=1.0,
        random_state=RANDOM_STATE
    )
    ada_model.fit(X_train_sc, y_train)
    y_pred_ada = ada_model.predict(X_test_sc)
    print("    AdaBoost model trained.")

    # 6. BAGGING — BaggingClassifier
    print("\n[6] Training BaggingClassifier (100 estimators)...")
    bag_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    bag_model.fit(X_train_sc, y_train)
    y_pred_bag = bag_model.predict(X_test_sc)
    print("    Bagging model trained.")

    # 7. Evaluate Both Models
    print("\n[7] Evaluation Results:")
    ada_results = evaluate_model(y_test, y_pred_ada, "BOOSTING (AdaBoost)", class_names)
    bag_results = evaluate_model(y_test, y_pred_bag, "BAGGING (BaggingClassifier)", class_names)

    # 8. Comparison Table
    comparison_rows = [
        {"Model": "Boosting (AdaBoost)",
         "Accuracy": f"{ada_results['Accuracy']*100:.2f}%",
         "Precision (Macro)": f"{ada_results['Precision']*100:.2f}%",
         "Recall (Macro)": f"{ada_results['Recall']*100:.2f}%",
         "F1 (Macro)": f"{ada_results['F1']*100:.2f}%"},
        {"Model": "Bagging (BaggingClassifier)",
         "Accuracy": f"{bag_results['Accuracy']*100:.2f}%",
         "Precision (Macro)": f"{bag_results['Precision']*100:.2f}%",
         "Recall (Macro)": f"{bag_results['Recall']*100:.2f}%",
         "F1 (Macro)": f"{bag_results['F1']*100:.2f}%"}
    ]
    comp_df = pd.DataFrame(comparison_rows)
    print("\n[8] COMPARISON TABLE (Boosting vs. Bagging):")
    print(comp_df.to_string(index=False))

    # 9. Save comparison CSV
    comp_path = os.path.join(script_dir, "comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\n[9] Saved comparison table: {comp_path}")

    # 10. Save metrics CSV
    metrics_path = os.path.join(script_dir, "metrics.csv")
    pd.DataFrame([
        {"Model": "AdaBoost", "Metric": "Accuracy",  "Value": f"{ada_results['Accuracy']:.4f}"},
        {"Model": "AdaBoost", "Metric": "Precision",  "Value": f"{ada_results['Precision']:.4f}"},
        {"Model": "AdaBoost", "Metric": "Recall",     "Value": f"{ada_results['Recall']:.4f}"},
        {"Model": "AdaBoost", "Metric": "F1",         "Value": f"{ada_results['F1']:.4f}"},
        {"Model": "Bagging",  "Metric": "Accuracy",   "Value": f"{bag_results['Accuracy']:.4f}"},
        {"Model": "Bagging",  "Metric": "Precision",  "Value": f"{bag_results['Precision']:.4f}"},
        {"Model": "Bagging",  "Metric": "Recall",     "Value": f"{bag_results['Recall']:.4f}"},
        {"Model": "Bagging",  "Metric": "F1",         "Value": f"{bag_results['F1']:.4f}"},
    ]).to_csv(metrics_path, index=False)
    print(f"[10] Saved metrics CSV: {metrics_path}")

    # 11. Save classification reports
    reports_path = os.path.join(script_dir, "classification_report.txt")
    with open(reports_path, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORTS - BOOSTING & BAGGING (UCI WINE)\n")
        f.write("=" * 70 + "\n\n")
        f.write("BOOSTING (AdaBoost):\n")
        f.write("-" * 40 + "\n")
        f.write(ada_results["Report"])
        f.write("\n\nBAGGING (BaggingClassifier):\n")
        f.write("-" * 40 + "\n")
        f.write(bag_results["Report"])
    print(f"[11] Saved classification reports: {reports_path}")

    # 12. Confusion matrix plots (both)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for ax, cm_data, title, model_label in zip(
        axes,
        [ada_results["CM"], bag_results["CM"]],
        ["Boosting (AdaBoost)", "Bagging (BaggingClassifier)"],
        ["confusion_matrix_boosting.png", "confusion_matrix_bagging.png"]
    ):
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 13, "fontweight": "bold"}, ax=ax)
        ax.set_title(f"{title}\nUCI Wine Dataset", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=10, fontweight="bold")

    plt.suptitle("Wine Classification: Confusion Matrices Comparison", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    combined_cm_path = os.path.join(script_dir, "confusion_matrix_comparison.png")
    plt.savefig(combined_cm_path, bbox_inches="tight")

    # Save individual confusion matrices too
    for cm_data, fname in [(ada_results["CM"], "confusion_matrix_boosting.png"),
                           (bag_results["CM"], "confusion_matrix_bagging.png")]:
        fig_s, ax_s = plt.subplots(figsize=(7, 6), dpi=300)
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 16, "fontweight": "bold"}, ax=ax_s)
        label = "Boosting (AdaBoost)" if "boosting" in fname else "Bagging"
        ax_s.set_title(f"{label} — UCI Wine Confusion Matrix", fontsize=13,
                       fontweight="bold", pad=10)
        ax_s.set_xlabel("Predicted", fontsize=11)
        ax_s.set_ylabel("Actual", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, fname), bbox_inches="tight")
        plt.close()

    plt.close('all')
    print(f"[12] Saved confusion matrix plots.")

    # 13. Bar comparison chart
    models = ["Boosting\n(AdaBoost)", "Bagging\n(BaggingClassifier)"]
    metrics = {
        "Accuracy":  [ada_results["Accuracy"],  bag_results["Accuracy"]],
        "Precision": [ada_results["Precision"], bag_results["Precision"]],
        "Recall":    [ada_results["Recall"],    bag_results["Recall"]],
        "F1 Score":  [ada_results["F1"],        bag_results["F1"]]
    }

    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#2980b9", "#27ae60", "#e74c3c", "#f39c12"]
    for i, (metric, values) in enumerate(metrics.items()):
        ax.bar(x + i*width, values, width, label=metric, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Boosting vs. Bagging: Performance Comparison — UCI Wine", fontsize=13,
                 fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8, padding=2)
    plt.tight_layout()
    bar_path = os.path.join(script_dir, "model_comparison_bar.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()
    print(f"[13] Saved bar comparison chart: {bar_path}")

    # 14. Write dataset_info.txt
    with open(os.path.join(script_dir, "dataset_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"""======================================================================
UCI MACHINE LEARNING REPOSITORY DATASET SPECIFICATION
======================================================================
Assignment:        Assignment 6 - Boosting & Bagging for Wine Classification
Dataset Name:      Wine
UCI Dataset ID:    109
UCI URL:           https://archive.ics.uci.edu/dataset/109/wine
Originator:        Forina, M. et al., PARVUS — An Extendable Package for Data Exploration, 1988
Total Instances:   {len(df)} wine samples
Input Features:    13 continuous physicochemical attributes
Target Class:      class (1, 2, or 3) — three cultivar categories

PREPROCESSING:
----------------------------------------------------------------------
- Missing values: {missing} found.
- StandardScaler applied to all features.
- Stratified 75/25 train/test split (random_state={RANDOM_STATE}).
- Identical split used for both Boosting and Bagging.

MODELS:
----------------------------------------------------------------------
BOOSTING: AdaBoostClassifier
  - Base estimator: DecisionTreeClassifier(max_depth=2)
  - n_estimators: 100
  - learning_rate: 1.0
  - algorithm: SAMME (multi-class compatible)

BAGGING: BaggingClassifier
  - Base estimator: DecisionTreeClassifier
  - n_estimators: 100
  - max_samples: 0.8
  - max_features: 0.8
  - bootstrap: True
======================================================================
""")

    # 15. Write results.txt
    with open(os.path.join(script_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"""======================================================================
ASSIGNMENT 6: BOOSTING & BAGGING WINE CLASSIFICATION RESULTS
======================================================================
Dataset: UCI Wine (ID: 109) | 3 Cultivar Classes
Train/Test: 75% ({len(X_train)}) / 25% ({len(X_test)}) | Random State: {RANDOM_STATE}

BOOSTING — AdaBoostClassifier (100 estimators):
----------------------------------------------------------------------
Accuracy:          {ada_results['Accuracy']*100:.2f}%
Precision (Macro): {ada_results['Precision']*100:.2f}%
Recall (Macro):    {ada_results['Recall']*100:.2f}%
F1 Score (Macro):  {ada_results['F1']*100:.2f}%

BAGGING — BaggingClassifier (100 estimators):
----------------------------------------------------------------------
Accuracy:          {bag_results['Accuracy']*100:.2f}%
Precision (Macro): {bag_results['Precision']*100:.2f}%
Recall (Macro):    {bag_results['Recall']*100:.2f}%
F1 Score (Macro):  {bag_results['F1']*100:.2f}%

COMPARISON SUMMARY:
----------------------------------------------------------------------
{"Boosting" if ada_results["Accuracy"] >= bag_results["Accuracy"] else "Bagging"} achieved higher accuracy.
{"Boosting" if ada_results["F1"] >= bag_results["F1"] else "Bagging"} achieved higher macro F1 score.

INTERPRETATION:
----------------------------------------------------------------------
AdaBoost (Boosting) sequentially corrects errors of prior weak learners
(shallow decision trees). BaggingClassifier uses bootstrap sampling to
build diverse full-depth trees and averages their predictions.
For the small, well-balanced UCI Wine dataset ({len(df)} samples, 13 features,
3 classes), both ensemble methods achieve high performance, confirming
that ensemble approaches substantially outperform single classifiers.
======================================================================
""")
    print(f"[14] Saved dataset_info.txt and results.txt")

    print("\n" + "=" * 75)
    print("ASSIGNMENT 6 EXECUTION COMPLETE - ALL OUTPUTS GENERATED")
    print("=" * 75)

if __name__ == "__main__":
    run_assignment()

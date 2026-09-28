"""
Assignment 7: K-Nearest Neighbors (KNN) for Iris Classification
Model: KNeighborsClassifier
Dataset: UCI Iris (UCI ID: 53)
Target: species (Iris-setosa, Iris-versicolor, Iris-virginica)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

def setup_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dataset_path = os.path.join(script_dir, "..", "datasets", "assignment_7", "iris.csv")
    return script_dir, os.path.abspath(local_dataset_path)

def load_data(dataset_path):
    if os.path.exists(dataset_path):
        print(f"[INFO] Loading dataset from local cache: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("[INFO] Fetching from UCI ML Repository (ID: 53)...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=53)
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
    print("ASSIGNMENT 7: K-NEAREST NEIGHBORS (KNN) FOR IRIS CLASSIFICATION")
    print("Dataset: UCI Iris (ID: 53)")
    print("Target: species (Setosa / Versicolor / Virginica)")
    print("=" * 75)

    # 1. Load Data
    df = load_data(dataset_path)
    print(f"\n[1] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    target_col = 'class' if 'class' in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"    Features: {feature_cols}")
    print(f"    Target: '{target_col}'")
    print(f"\nClass Distribution:")
    print(df[target_col].value_counts().to_string())

    # 2. Missing values
    missing = df.isnull().sum().sum()
    print(f"\n[2] Missing Values: {missing}")
    if missing > 0:
        df = df.dropna()

    X = df[feature_cols].values
    y = df[target_col].values
    class_names = sorted(df[target_col].unique())

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[3] Train/Test Split (70/30): {len(X_train)} train | {len(X_test)} test")

    # 4. Feature Scaling (critical for KNN — distance-based)
    print("\n[4] Applying StandardScaler (critical for distance-based KNN)...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 5. Select best K using cross-validation on training set
    k_range = range(1, 21)
    cv_scores = []
    print("\n[5] Evaluating K values (1-20) via 5-Fold Cross-Validation on Training Set:")
    for k in k_range:
        knn_cv = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_cv, X_train_sc, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
        print(f"    K={k:2d}: CV Accuracy = {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    best_k = k_range[np.argmax(cv_scores)]
    best_cv_acc = max(cv_scores)
    print(f"\n    Best K: {best_k} (CV Accuracy: {best_cv_acc*100:.2f}%)")

    # 6. Plot Accuracy vs K
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(list(k_range), [s*100 for s in cv_scores], 'o-', color='#2980b9',
             markersize=7, linewidth=2, markerfacecolor='white', markeredgewidth=2)
    plt.axvline(x=best_k, color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Best K = {best_k} (Acc = {best_cv_acc*100:.2f}%)')
    plt.title("KNN: Accuracy vs. K (5-Fold CV) — UCI Iris Dataset", fontsize=13,
              fontweight="bold", pad=10)
    plt.xlabel("Number of Neighbors (K)", fontsize=12)
    plt.ylabel("Cross-Validation Accuracy (%)", fontsize=12)
    plt.xticks(list(k_range))
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    k_plot_path = os.path.join(script_dir, "accuracy_vs_k.png")
    plt.savefig(k_plot_path, bbox_inches="tight")
    plt.close()
    print(f"\n[6] Saved Accuracy vs K plot: {k_plot_path}")

    # 7. Train Final Model with Best K
    print(f"\n[7] Training Final KNN Model with K={best_k}...")
    knn_final = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn_final.fit(X_train_sc, y_train)
    y_pred = knn_final.predict(X_test_sc)

    # 8. Evaluation
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n[8] Final Model Evaluation (K={best_k}, Test Set):")
    print(f"    Accuracy:          {acc*100:.2f}% ({acc:.4f})")
    print(f"    Precision (Macro): {prec*100:.2f}% ({prec:.4f})")
    print(f"    Recall (Macro):    {rec*100:.2f}% ({rec:.4f})")
    print(f"    F1 Score (Macro):  {f1*100:.2f}% ({f1:.4f})")

    cls_report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(f"\nClassification Report:\n{cls_report}")

    # 9. Confusion Matrix Plot
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "fontweight": "bold"})
    plt.title(f"KNN Confusion Matrix (K={best_k}) — UCI Iris", fontsize=13,
              fontweight="bold", pad=12)
    plt.xlabel("Predicted Species", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Species", fontsize=11, fontweight="bold")
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    cm_path = os.path.join(script_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[9] Saved confusion matrix: {cm_path}")

    # 10. Save classification report
    report_path = os.path.join(script_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"CLASSIFICATION REPORT - KNN K={best_k} (UCI IRIS)\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_report)
    print(f"[10] Saved classification report: {report_path}")

    # 11. Save metrics CSV
    metrics_path = os.path.join(script_dir, "metrics.csv")
    k_cv_rows = [{"Metric": f"CV_Acc_K{k}", "Value": f"{s:.4f}"}
                 for k, s in zip(k_range, cv_scores)]
    summary_rows = [
        {"Metric": "Best_K", "Value": str(best_k)},
        {"Metric": "Accuracy", "Value": f"{acc:.4f}"},
        {"Metric": "Precision_Macro", "Value": f"{prec:.4f}"},
        {"Metric": "Recall_Macro", "Value": f"{rec:.4f}"},
        {"Metric": "F1_Macro", "Value": f"{f1:.4f}"},
    ]
    pd.DataFrame(summary_rows + k_cv_rows).to_csv(metrics_path, index=False)
    print(f"[11] Saved metrics CSV: {metrics_path}")

    # 12. Write dataset_info.txt
    with open(os.path.join(script_dir, "dataset_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"""======================================================================
UCI MACHINE LEARNING REPOSITORY DATASET SPECIFICATION
======================================================================
Assignment:        Assignment 7 - KNN for Iris Classification
Dataset Name:      Iris
UCI Dataset ID:    53
UCI URL:           https://archive.ics.uci.edu/dataset/53/iris
Originator:        R.A. Fisher, 1936
Total Instances:   {len(df)} flower samples (50 per class)
Input Features:    4 continuous morphological measurements
  - sepal length (cm)
  - sepal width (cm)
  - petal length (cm)
  - petal width (cm)
Target Classes:    Iris-setosa, Iris-versicolor, Iris-virginica

PREPROCESSING:
----------------------------------------------------------------------
- Missing values: {missing} found.
- StandardScaler applied (required for Euclidean distance in KNN).
- 70/30 stratified train/test split (random_state={RANDOM_STATE}).
- K selected via 5-fold cross-validation over K=1..20.
- Optimal K = {best_k} (CV Accuracy = {best_cv_acc*100:.2f}%)
======================================================================
""")

    # 13. Write results.txt
    with open(os.path.join(script_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"""======================================================================
ASSIGNMENT 7: KNN IRIS CLASSIFICATION RESULTS REPORT
======================================================================
Algorithm:         K-Nearest Neighbors (KNeighborsClassifier)
Distance Metric:   Euclidean
Dataset:           UCI Iris (ID: 53)
Train/Test Split:  70% ({len(X_train)}) / 30% ({len(X_test)}) | Stratified
Random State:      {RANDOM_STATE}
K Selection:       5-fold CV over K=1..20 → Best K = {best_k}

FINAL METRICS (TEST SET, K={best_k}):
----------------------------------------------------------------------
Accuracy:          {acc*100:.2f}%
Precision (Macro): {prec*100:.2f}%
Recall (Macro):    {rec*100:.2f}%
F1 Score (Macro):  {f1*100:.2f}%

INTERPRETATION:
----------------------------------------------------------------------
KNN with K={best_k} achieves {acc*100:.2f}% accuracy on the Iris dataset.
As a distance-based algorithm, KNN is highly sensitive to feature scale
differences; StandardScaler was therefore applied to equalise feature
contributions. Cross-validation across K=1 to 20 confirmed K={best_k}
as optimal — small K values overfit training samples, while large K
values introduce bias. Iris-setosa is trivially linearly separable,
while Iris-versicolor and Iris-virginica have some class overlap.
======================================================================
""")
    print(f"[12] Saved dataset_info.txt and results.txt")

    print("\n" + "=" * 75)
    print("ASSIGNMENT 7 EXECUTION COMPLETE - ALL OUTPUTS GENERATED")
    print("=" * 75)

if __name__ == "__main__":
    run_assignment()

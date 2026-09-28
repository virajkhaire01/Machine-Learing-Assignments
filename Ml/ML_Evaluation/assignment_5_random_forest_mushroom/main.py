"""
Assignment 5: Random Forest for Mushroom Classification
Model: RandomForestClassifier
Dataset: UCI Mushroom (UCI ID: 73)
Target: class (p = Poisonous, e = Edible)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

def setup_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dataset_path = os.path.join(script_dir, "..", "datasets", "assignment_5", "mushroom.csv")
    return script_dir, os.path.abspath(local_dataset_path)

def load_data(dataset_path):
    if os.path.exists(dataset_path):
        print(f"[INFO] Loading dataset from local cache: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("[INFO] Fetching from UCI ML Repository (ID: 73)...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=73)
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
    print("ASSIGNMENT 5: RANDOM FOREST FOR MUSHROOM CLASSIFICATION")
    print("Dataset: UCI Mushroom (ID: 73)")
    print("Target: Edible (e) vs Poisonous (p)")
    print("=" * 75)

    # 1. Load Data
    df = load_data(dataset_path)
    print(f"\n[1] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Determine target column
    target_col = 'class' if 'class' in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"    Target column: '{target_col}'")
    print(f"    Feature count: {len(feature_cols)}")
    print(f"\nClass Distribution:")
    print(df[target_col].value_counts().to_string())

    # 2. Handle Missing Values
    # In mushroom dataset 'stalk-root' has '?' as missing values
    print(f"\n[2] Handling Missing Values...")
    initial_rows = len(df)
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    missing_per_col = df.isnull().sum()
    cols_with_missing = missing_per_col[missing_per_col > 0]
    if len(cols_with_missing) > 0:
        print(f"    Columns with missing values:")
        for col, cnt in cols_with_missing.items():
            print(f"      '{col}': {cnt} missing ({cnt/len(df)*100:.1f}%)")
        # Fill with mode (most frequent category) per column
        for col in cols_with_missing.index:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
        print(f"    Strategy: Filled with column mode (most frequent category).")
    else:
        print(f"    No missing values found after '?' check.")

    # 3. Encode Categorical Features
    print(f"\n[3] Encoding Categorical Features using LabelEncoder...")
    le_dict = {}
    df_encoded = df.copy()
    for col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le

    # Encode target separately for label names
    target_classes = le_dict[target_col].classes_
    print(f"    Target classes encoded: {dict(enumerate(target_classes))}")
    print(f"    All {len(feature_cols)} feature columns label-encoded.")

    X = df_encoded[feature_cols].values
    y = df_encoded[target_col].values

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[4] Train/Test Split: {len(X_train)} train | {len(X_test)} test")

    # 5. Train Random Forest
    print("\n[5] Training RandomForestClassifier (100 trees)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print(f"    Model trained with {rf_model.n_estimators} decision trees.")

    # 6. Prediction & Evaluation
    y_pred = rf_model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec  = recall_score(y_test, y_pred, average='binary')
    f1   = f1_score(y_test, y_pred, average='binary')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"\n[6] Evaluation Metrics:")
    print(f"    Accuracy:              {acc*100:.2f}% ({acc:.4f})")
    print(f"    Precision (Poisonous): {prec*100:.2f}% ({prec:.4f})")
    print(f"    Recall (Poisonous):    {rec*100:.2f}% ({rec:.4f})")
    print(f"    F1 Score (Poisonous):  {f1*100:.2f}% ({f1:.4f})")
    print(f"    Macro F1:              {f1_macro:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n    Confusion Matrix (rows=Actual, cols=Predicted):")
    class_name_list = list(target_classes)
    print(pd.DataFrame(cm, index=class_name_list, columns=class_name_list).to_string())

    cls_report = classification_report(y_test, y_pred,
                                       target_names=class_name_list, digits=4)
    print("\nClassification Report:")
    print(cls_report)

    # 7. Save classification report
    report_path = os.path.join(script_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT - RANDOM FOREST (UCI MUSHROOM)\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_report)
    print(f"[7] Saved classification report: {report_path}")

    # 8. Confusion Matrix Plot
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=class_name_list,
                yticklabels=class_name_list,
                annot_kws={"size": 16, "fontweight": "bold"})
    plt.title("Random Forest Confusion Matrix — UCI Mushroom", fontsize=13,
              fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    cm_path = os.path.join(script_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[8] Saved confusion matrix: {cm_path}")

    # 9. Feature Importance
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(20)

    print("\nTop 10 Most Important Features:")
    print(fi_df.head(10).to_string(index=False))

    plt.figure(figsize=(10, 7), dpi=300)
    palette = sns.color_palette("viridis", len(fi_df))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette=palette)
    plt.title("Random Forest Feature Importances (Top 20) — UCI Mushroom", fontsize=13,
              fontweight="bold", pad=10)
    plt.xlabel("Mean Decrease in Impurity (Importance)", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    fi_path = os.path.join(script_dir, "feature_importance.png")
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    print(f"[9] Saved feature importance chart: {fi_path}")

    # 10. Save metrics CSV
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    metrics_path = os.path.join(script_dir, "metrics.csv")
    pd.DataFrame([
        {"Metric": "Accuracy", "Value": f"{acc:.4f}"},
        {"Metric": "Precision_Poisonous", "Value": f"{prec:.4f}"},
        {"Metric": "Recall_Poisonous", "Value": f"{rec:.4f}"},
        {"Metric": "F1_Poisonous", "Value": f"{f1:.4f}"},
        {"Metric": "Macro_F1", "Value": f"{f1_macro:.4f}"},
        {"Metric": "True_Negatives", "Value": str(tn)},
        {"Metric": "False_Positives", "Value": str(fp)},
        {"Metric": "False_Negatives", "Value": str(fn)},
        {"Metric": "True_Positives", "Value": str(tp)},
        {"Metric": "N_Estimators", "Value": "100"},
        {"Metric": "Top_Feature", "Value": fi_df.iloc[0]["Feature"]}
    ]).to_csv(metrics_path, index=False)
    print(f"\n[10] Saved metrics CSV: {metrics_path}")

    # 11. Write dataset_info.txt
    with open(os.path.join(script_dir, "dataset_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"""======================================================================
UCI MACHINE LEARNING REPOSITORY DATASET SPECIFICATION
======================================================================
Assignment:        Assignment 5 - Random Forest for Mushroom Classification
Dataset Name:      Mushroom
UCI Dataset ID:    73
UCI URL:           https://archive.ics.uci.edu/dataset/73/mushroom
Total Instances:   {len(df)} mushroom samples
Input Features:    {len(feature_cols)} categorical morphological attributes
Target Class:      class (e = Edible, p = Poisonous)

ENCODING APPLIED:
----------------------------------------------------------------------
- All 22 categorical feature columns encoded with LabelEncoder.
- Missing values ('?') in 'stalk-root': {cols_with_missing.get('stalk-root', 0)} instances filled with mode.
- Train/Test Split: 80% ({len(X_train)}) / 20% ({len(X_test)}), stratified.
- No feature scaling needed (Random Forests are scale-invariant).
======================================================================
""")

    # 12. Write results.txt
    with open(os.path.join(script_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"""======================================================================
ASSIGNMENT 5: RANDOM FOREST MUSHROOM CLASSIFICATION RESULTS REPORT
======================================================================
Algorithm:         RandomForestClassifier
Estimators:        100 trees
Dataset:           UCI Mushroom (ID: 73)
Train/Test Split:  80% ({len(X_train)}) / 20% ({len(X_test)})
Random State:      {RANDOM_STATE}

FINAL METRICS (TEST SET):
----------------------------------------------------------------------
Accuracy:              {acc*100:.2f}%
Precision (Poisonous): {prec*100:.2f}%
Recall (Poisonous):    {rec*100:.2f}%
F1 Score (Poisonous):  {f1*100:.2f}%
Macro F1:              {f1_macro*100:.2f}%
Top Feature:           {fi_df.iloc[0]["Feature"]} (Importance: {fi_df.iloc[0]["Importance"]:.4f})

INTERPRETATION:
----------------------------------------------------------------------
The Random Forest model achieved {acc*100:.2f}% accuracy classifying mushrooms
as edible or poisonous from 22 categorical morphological features.
The most discriminative feature is '{fi_df.iloc[0]["Feature"]}', which
Random Forest consistently identifies as the strongest predictor of
mushroom toxicity. High recall ({rec*100:.2f}%) is critically important here:
missing a poisonous mushroom (False Negative) would be life-threatening.
The ensemble of {rf_model.n_estimators} trees significantly reduces variance compared
to a single decision tree and achieves near-perfect classification.
======================================================================
""")
    print(f"[11] Saved dataset_info.txt and results.txt")

    print("\n" + "=" * 75)
    print("ASSIGNMENT 5 EXECUTION COMPLETE - ALL OUTPUTS GENERATED")
    print("=" * 75)

if __name__ == "__main__":
    run_assignment()

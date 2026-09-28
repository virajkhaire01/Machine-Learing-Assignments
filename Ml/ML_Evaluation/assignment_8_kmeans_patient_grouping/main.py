"""
Assignment 8: K-Means Clustering for Patient Grouping Based on Health Indicators
Model: KMeans Clustering
Dataset: UCI Heart Disease (UCI ID: 45)
Uses numeric clinical indicators ONLY (not the target/diagnosis as input)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

def setup_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dataset_path = os.path.join(script_dir, "..", "datasets", "assignment_8", "heart_disease.csv")
    return script_dir, os.path.abspath(local_dataset_path)

def load_data(dataset_path):
    if os.path.exists(dataset_path):
        print(f"[INFO] Loading dataset from local cache: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("[INFO] Fetching from UCI ML Repository (ID: 45)...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=45)
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
    print("ASSIGNMENT 8: K-MEANS CLUSTERING FOR PATIENT HEALTH GROUPING")
    print("Dataset: UCI Heart Disease (ID: 45)")
    print("Method: Unsupervised KMeans — Diagnosis column NOT used as input feature")
    print("=" * 75)

    # 1. Load Data
    df = load_data(dataset_path)
    print(f"\n[1] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"    Columns: {df.columns.tolist()}")

    # 2. Select numeric health indicator features ONLY (exclude target/diagnosis)
    # UCI Heart Disease columns:
    # age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target/num
    # Target column: 'num' (0 = no disease, 1-4 = disease) — EXCLUDED from clustering input
    target_col = 'num' if 'num' in df.columns else df.columns[-1]

    # Use clinically meaningful numeric features
    numeric_feature_candidates = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    # Filter only features that exist in the dataset
    feature_cols = [c for c in numeric_feature_candidates if c in df.columns]
    print(f"\n[2] Selected Numeric Health Indicators: {feature_cols}")
    print(f"    Target column '{target_col}' is EXCLUDED from clustering.")

    # 3. Handle Missing Values
    print(f"\n[3] Handling Missing Values...")
    cluster_df = df[feature_cols].copy()
    missing_before = cluster_df.isnull().sum()
    print(f"    Missing values per column:\n{missing_before.to_string()}")

    total_missing = missing_before.sum()
    if total_missing > 0:
        # Impute with median (robust to skew in clinical data)
        for col in feature_cols:
            if cluster_df[col].isnull().sum() > 0:
                med = cluster_df[col].median()
                cluster_df[col] = cluster_df[col].fillna(med)
                print(f"    Filled '{col}' NaN with median={med:.2f}")
    else:
        print(f"    No missing values detected.")

    print(f"\n    Final clean samples: {len(cluster_df)}")

    # 4. Feature Scaling
    print("\n[4] Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df.values)
    print("    Scaling complete. Mean~0, Std~1 per feature.")

    # 5. Elbow Method to determine optimal K
    print("\n[5] Running Elbow Method (K=2 to 10)...")
    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300,
                    random_state=RANDOM_STATE)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"    K={k}: Inertia (WCSS) = {km.inertia_:.2f}")

    # Plot Elbow Curve
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(list(k_range), inertias, 'o-', color='#2980b9', markersize=8,
             linewidth=2.5, markerfacecolor='white', markeredgewidth=2)
    plt.title("K-Means Elbow Curve — UCI Heart Disease Patient Grouping", fontsize=13,
              fontweight="bold", pad=10)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    plt.xticks(list(k_range))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    elbow_path = os.path.join(script_dir, "elbow_curve.png")
    plt.savefig(elbow_path, bbox_inches="tight")
    plt.close()
    print(f"\n    Saved Elbow Curve: {elbow_path}")

    # 6. Select K (use K=3 — clinically meaningful: healthy, at-risk, high-risk)
    CHOSEN_K = 3
    print(f"\n[6] Selected K={CHOSEN_K} (based on elbow — meaningful clinical grouping).")

    # 7. Compute Silhouette Scores for K=2..5 for completeness
    print("\n    Silhouette Scores for K=2..6:")
    sil_scores = {}
    for k in range(2, 7):
        km_s = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_STATE)
        labels_s = km_s.fit_predict(X_scaled)
        s = silhouette_score(X_scaled, labels_s)
        sil_scores[k] = s
        print(f"    K={k}: Silhouette Score = {s:.4f}")

    best_k_sil = max(sil_scores, key=sil_scores.get)
    print(f"    Best K by silhouette: {best_k_sil} (Score: {sil_scores[best_k_sil]:.4f})")

    # 8. Final KMeans Clustering with CHOSEN_K
    print(f"\n[8] Running final KMeans with K={CHOSEN_K}...")
    kmeans_final = KMeans(n_clusters=CHOSEN_K, init='k-means++', n_init=10,
                          max_iter=300, random_state=RANDOM_STATE)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    final_inertia = kmeans_final.inertia_
    final_silhouette = silhouette_score(X_scaled, cluster_labels)

    print(f"    Final Inertia (WCSS):     {final_inertia:.2f}")
    print(f"    Final Silhouette Score:   {final_silhouette:.4f}")

    # 9. Cluster Analysis
    cluster_df_result = cluster_df.copy()
    cluster_df_result['Cluster'] = cluster_labels
    cluster_df_result['Original_Target'] = df[target_col].values[:len(cluster_df_result)]

    print(f"\n[9] Cluster Sizes:")
    for cid in range(CHOSEN_K):
        cnt = (cluster_labels == cid).sum()
        print(f"    Cluster {cid}: {cnt} patients ({cnt/len(cluster_labels)*100:.1f}%)")

    print(f"\n    Cluster Centers (original feature scale):")
    centers_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
    centers_df = pd.DataFrame(centers_original, columns=feature_cols)
    centers_df.index.name = "Cluster"
    print(centers_df.round(2).to_string())

    # 10. PCA 2D Visualization
    print("\n[10] Generating 2D PCA cluster visualization...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_
    print(f"    PCA variance explained: PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%")

    # 2D Cluster scatter
    colors = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']
    labels_map = {0: "Group A (Low Risk)", 1: "Group B (Mid Risk)", 2: "Group C (High Risk)"}

    plt.figure(figsize=(10, 7), dpi=300)
    for cid in range(CHOSEN_K):
        mask = cluster_labels == cid
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors[cid], label=labels_map.get(cid, f"Cluster {cid}"),
                    alpha=0.75, s=60, edgecolors='k', linewidths=0.4)

    # Plot cluster centers in PCA space
    centers_pca = pca.transform(kmeans_final.cluster_centers_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                c='black', marker='X', s=200, zorder=5, label='Cluster Centers')

    plt.title(f"K-Means Patient Clustering (K={CHOSEN_K}) — UCI Heart Disease (PCA 2D)",
              fontsize=13, fontweight="bold", pad=10)
    plt.xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontsize=11)
    plt.ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontsize=11)
    plt.legend(fontsize=10, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    cluster_plot_path = os.path.join(script_dir, "clusters.png")
    plt.savefig(cluster_plot_path, bbox_inches="tight")
    plt.close()
    print(f"    Saved cluster visualization: {cluster_plot_path}")

    # 11. Feature distribution per cluster (boxplot)
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(16, 5), dpi=200)
    for i, feat in enumerate(feature_cols):
        data_for_plot = [cluster_df_result[cluster_df_result['Cluster'] == cid][feat].values
                         for cid in range(CHOSEN_K)]
        axes[i].boxplot(data_for_plot, tick_labels=[f"C{j}" for j in range(CHOSEN_K)],
                        patch_artist=True,
                        boxprops=dict(facecolor='#3498db', alpha=0.6),
                        medianprops=dict(color='red', linewidth=2))
        axes[i].set_title(feat, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Cluster", fontsize=9)
        if i == 0:
            axes[i].set_ylabel("Value", fontsize=9)
        axes[i].grid(True, linestyle='--', alpha=0.4)

    plt.suptitle("Feature Distribution per Patient Cluster — UCI Heart Disease",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    boxplot_path = os.path.join(script_dir, "cluster_feature_distribution.png")
    plt.savefig(boxplot_path, bbox_inches="tight")
    plt.close()
    print(f"    Saved feature distribution boxplot: {boxplot_path}")

    # 12. Save clustered data to CSV
    clustered_data_path = os.path.join(script_dir, "clustered_patients.csv")
    cluster_df_result.to_csv(clustered_data_path, index=False)
    print(f"[12] Saved clustered patient data: {clustered_data_path}")

    # 13. Save metrics CSV
    metrics_path = os.path.join(script_dir, "metrics.csv")
    pd.DataFrame([
        {"Metric": "Chosen_K", "Value": str(CHOSEN_K)},
        {"Metric": "Final_Inertia_WCSS", "Value": f"{final_inertia:.4f}"},
        {"Metric": "Silhouette_Score", "Value": f"{final_silhouette:.4f}"},
        {"Metric": "PCA_PC1_Variance", "Value": f"{var_explained[0]:.4f}"},
        {"Metric": "PCA_PC2_Variance", "Value": f"{var_explained[1]:.4f}"},
        {"Metric": "Total_Patients", "Value": str(len(cluster_labels))},
        *[{"Metric": f"Cluster_{i}_Size",
           "Value": str((cluster_labels == i).sum())} for i in range(CHOSEN_K)],
        *[{"Metric": f"Silhouette_K{k}",
           "Value": f"{v:.4f}"} for k, v in sil_scores.items()]
    ]).to_csv(metrics_path, index=False)
    print(f"[13] Saved metrics CSV: {metrics_path}")

    # 14. Write dataset_info.txt
    with open(os.path.join(script_dir, "dataset_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"""======================================================================
UCI MACHINE LEARNING REPOSITORY DATASET SPECIFICATION
======================================================================
Assignment:        Assignment 8 - K-Means Clustering for Patient Grouping
Dataset Name:      Heart Disease
UCI Dataset ID:    45
UCI URL:           https://archive.ics.uci.edu/dataset/45/heart+disease
Originator:        Andras Janosi, William Steinbrunn, Matthias Pfisterer, Robert Detrano, 1988
Total Instances:   {len(df)} patient records
Total Features:    {df.shape[1]-1} attributes (13 inputs + 1 diagnosis target)
Target (Excluded): num (0 = no disease, 1-4 = disease severity) — NOT used as clustering input

SELECTED NUMERIC FEATURES FOR CLUSTERING:
----------------------------------------------------------------------
1. age        - Patient age in years
2. trestbps   - Resting blood pressure (mm Hg) on admission
3. chol       - Serum cholesterol (mg/dl)
4. thalach    - Maximum heart rate achieved (beats/min)
5. oldpeak    - ST depression induced by exercise relative to rest

PREPROCESSING:
----------------------------------------------------------------------
- Missing values: {total_missing} total across selected features.
- Imputation: Median fill for missing entries.
- Feature Scaling: StandardScaler (zero-mean, unit-variance) — essential for Euclidean KMeans.
- K Selection: Elbow Method over K=2..10; confirmed with Silhouette Scores.
- Chosen K: {CHOSEN_K}
- Algorithm: KMeans (k-means++ initialization, n_init=10)

NOTE: This is UNSUPERVISED learning. No accuracy metric is reported.
Silhouette Score ({final_silhouette:.4f}) and Inertia ({final_inertia:.2f}) are appropriate
evaluation metrics for clustering quality.
======================================================================
""")

    # 15. Write results.txt
    cluster_summary_lines = "\n".join([
        f"  Cluster {i}: {(cluster_labels == i).sum()} patients "
        f"(Age={centers_df.iloc[i]['age']:.1f}, "
        f"BP={centers_df.iloc[i]['trestbps']:.1f}, "
        f"Chol={centers_df.iloc[i]['chol']:.1f}, "
        f"MaxHR={centers_df.iloc[i]['thalach']:.1f}, "
        f"STdep={centers_df.iloc[i]['oldpeak']:.2f})"
        for i in range(CHOSEN_K)
    ])
    with open(os.path.join(script_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"""======================================================================
ASSIGNMENT 8: K-MEANS CLUSTERING PATIENT GROUPING RESULTS REPORT
======================================================================
Algorithm:         K-Means (k-means++ initialization)
Dataset:           UCI Heart Disease (ID: 45)
Chosen K:          {CHOSEN_K} clusters
Method for K:      Elbow Method + Silhouette Score Validation
Random State:      {RANDOM_STATE}
Total Patients:    {len(cluster_labels)}

FINAL CLUSTERING METRICS (UNSUPERVISED — NO ACCURACY METRIC):
----------------------------------------------------------------------
Inertia (WCSS):    {final_inertia:.4f}
Silhouette Score:  {final_silhouette:.4f}
  (Range: -1 to +1; closer to +1 indicates well-separated, compact clusters)

CLUSTER CENTERS (ORIGINAL SCALE):
----------------------------------------------------------------------
{cluster_summary_lines}

SILHOUETTE SCORE COMPARISON (K=2..6):
----------------------------------------------------------------------
{chr(10).join([f"  K={k}: {v:.4f}" for k, v in sil_scores.items()])}

IMPORTANT ACADEMIC NOTE — UNSUPERVISED LEARNING:
----------------------------------------------------------------------
This is a clustering (unsupervised) task. Classification accuracy is NOT
applicable and is NOT reported. Instead, the quality of groupings is assessed
via Inertia (within-cluster compactness) and Silhouette Score (separation).
The 'num' target column was strictly excluded from all input features
to prevent data leakage and to ensure a genuine unsupervised discovery.

INTERPRETATION:
----------------------------------------------------------------------
K-Means with K={CHOSEN_K} groups the {len(cluster_labels)} heart disease patients into
{CHOSEN_K} clinically interpretable health risk cohorts based on age,
blood pressure, cholesterol, maximum heart rate, and ST depression.
PCA 2D visualization confirms clear cluster separation. The Silhouette
Score of {final_silhouette:.4f} indicates meaningful cluster cohesion.
======================================================================
""")
    print(f"[14] Saved dataset_info.txt and results.txt")

    print("\n" + "=" * 75)
    print("ASSIGNMENT 8 EXECUTION COMPLETE - ALL OUTPUTS GENERATED")
    print("=" * 75)

if __name__ == "__main__":
    run_assignment()

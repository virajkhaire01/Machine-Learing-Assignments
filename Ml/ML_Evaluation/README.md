# ML Evaluation — College Practical Submission

A complete, fully-executable collection of 8 Machine Learning assignments
implemented in Python using genuine UCI Machine Learning Repository datasets.
Every assignment has been executed, verified, and outputs saved automatically.

---

## Project Overview

This project was built as a B.Tech Machine Learning practical submission.
Each assignment implements a distinct ML algorithm on a real UCI dataset,
generates evaluation metrics, saves plots, and writes result files.

---

## Project Structure

```
ML_Evaluation/
├── README.md
├── requirements.txt
├── run_all.py                          ← Runs all 8 assignments
├── download_all_datasets.py            ← Downloads/caches all UCI datasets
├── MASTER_RESULTS.txt                  ← Generated after run_all.py
├── MASTER_METRICS.csv                  ← Generated after run_all.py
│
├── datasets/
│   ├── assignment_1/   auto_mpg.csv
│   ├── assignment_2/   real_estate_valuation.csv
│   ├── assignment_3/   spambase.csv
│   ├── assignment_4/   spambase.csv
│   ├── assignment_5/   mushroom.csv
│   ├── assignment_6/   wine.csv
│   ├── assignment_7/   iris.csv
│   └── assignment_8/   heart_disease.csv
│
├── assignment_1_simple_linear_regression/
├── assignment_2_multiple_linear_regression/
├── assignment_3_svm_email/
├── assignment_4_decision_tree_spam/
├── assignment_5_random_forest_mushroom/
├── assignment_6_boosting_bagging_wine/
├── assignment_7_knn_iris/
└── assignment_8_kmeans_patient_grouping/
```

Each assignment folder contains:
- `main.py` — Fully runnable Python script
- `dataset_info.txt` — Dataset audit and preprocessing documentation
- `results.txt` — Final metrics and interpretation
- `metrics.csv` — Machine-readable metrics table
- `execution_log.txt` — Captured console output from execution
- `predictions.csv` — (where applicable) Actual vs predicted values
- `classification_report.txt` — (classification tasks) Detailed per-class report
- `confusion_matrix.png` — (classification tasks) Heatmap visualization
- `plots/` — Regression plots (assignments 1–2)

---

## Assignment List

| # | Title | Dataset | Algorithm |
|---|-------|---------|-----------|
| 1 | Simple Linear Regression | UCI Auto MPG (ID: 9) | LinearRegression (OLS) |
| 2 | Multiple Linear Regression | UCI Real Estate Valuation (ID: 477) | LinearRegression (OLS) |
| 3 | SVM for Email Classification | UCI Spambase (ID: 94) | SVC (RBF kernel) |
| 4 | Decision Tree for Spam Classification | UCI Spambase (ID: 94) | DecisionTreeClassifier |
| 5 | Random Forest for Mushroom Classification | UCI Mushroom (ID: 73) | RandomForestClassifier |
| 6 | Boosting and Bagging for Wine Classification | UCI Wine (ID: 109) | AdaBoost + BaggingClassifier |
| 7 | KNN for Iris Classification | UCI Iris (ID: 53) | KNeighborsClassifier |
| 8 | K-Means Clustering for Patient Grouping | UCI Heart Disease (ID: 45) | KMeans (k-means++) |

---

## Datasets Used

| Assignment | Dataset Name | UCI ID | URL |
|---|---|---|---|
| 1 | Auto MPG | 9 | https://archive.ics.uci.edu/dataset/9/auto+mpg |
| 2 | Real Estate Valuation | 477 | https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set |
| 3, 4 | Spambase | 94 | https://archive.ics.uci.edu/dataset/94/spambase |
| 5 | Mushroom | 73 | https://archive.ics.uci.edu/dataset/73/mushroom |
| 6 | Wine | 109 | https://archive.ics.uci.edu/dataset/109/wine |
| 7 | Iris | 53 | https://archive.ics.uci.edu/dataset/53/iris |
| 8 | Heart Disease | 45 | https://archive.ics.uci.edu/dataset/45/heart+disease |

---

## Algorithms Used

| Assignment | Algorithm | Key Parameters |
|---|---|---|
| 1 | Simple Linear Regression | 1 feature (displacement → mpg) |
| 2 | Multiple Linear Regression | 6 features → house price |
| 3 | SVM (Support Vector Machine) | RBF kernel, C=1.0, StandardScaler |
| 4 | Decision Tree | Gini, max_depth=8, min_samples_leaf=2 |
| 5 | Random Forest | 100 trees, max_features='sqrt' |
| 6 | AdaBoost (Boosting) | 100 estimators, base: DecisionTree(depth=2) |
| 6 | Bagging | 100 estimators, 80% sampling |
| 7 | K-Nearest Neighbors | Best K via 5-fold CV (K=14 chosen), Euclidean |
| 8 | K-Means Clustering | K=3, k-means++, elbow + silhouette |

---

## How to Install Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `ucimlrepo`
- `scipy`

---

## How to Download Datasets

Datasets are automatically fetched from UCI when you run each assignment.
To pre-download all datasets into the `datasets/` folder:

```bash
python download_all_datasets.py
```

---

## How to Run Individual Assignments

Navigate to the project root and run:

```bash
# Assignment 1 — Simple Linear Regression
python assignment_1_simple_linear_regression/main.py

# Assignment 2 — Multiple Linear Regression
python assignment_2_multiple_linear_regression/main.py

# Assignment 3 — SVM Email Classification
python assignment_3_svm_email/main.py

# Assignment 4 — Decision Tree Spam Classification
python assignment_4_decision_tree_spam/main.py

# Assignment 5 — Random Forest Mushroom Classification
python assignment_5_random_forest_mushroom/main.py

# Assignment 6 — Boosting and Bagging Wine Classification
python assignment_6_boosting_bagging_wine/main.py

# Assignment 7 — KNN Iris Classification
python assignment_7_knn_iris/main.py

# Assignment 8 — K-Means Patient Clustering
python assignment_8_kmeans_patient_grouping/main.py
```

---

## How to Run All Assignments

```bash
python run_all.py
```

This runs all 8 assignments sequentially and generates:
- `MASTER_RESULTS.txt` — PASS/FAIL summary for each assignment
- `MASTER_METRICS.csv` — All key metrics in one CSV

---

## Output Locations

Each assignment stores its output inside its own folder:

| File | Description |
|---|---|
| `results.txt` | Final interpretation and all metrics |
| `metrics.csv` | Metrics in CSV format |
| `execution_log.txt` | Full console output from execution |
| `dataset_info.txt` | Dataset audit and preprocessing steps |
| `predictions.csv` | Actual vs predicted values (regression/classification) |
| `classification_report.txt` | Per-class precision/recall/F1 |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `plots/` | Regression line, residuals (assignments 1–2) |
| `tree.png` | Decision tree visualization (assignment 4) |
| `feature_importance.png` | Feature importance chart (assignments 4, 5) |
| `accuracy_vs_k.png` | KNN K-selection chart (assignment 7) |
| `elbow_curve.png` | KMeans elbow curve (assignment 8) |
| `clusters.png` | PCA 2D cluster visualization (assignment 8) |

---

## Key Results Summary

| Assignment | Metric | Score |
|---|---|---|
| 1 — Linear Regression | R² | 0.6633 |
| 2 — Multiple Regression | R² | 0.6811 |
| 3 — SVM | Accuracy | 92.73% |
| 4 — Decision Tree | Accuracy | 90.12% |
| 5 — Random Forest | Accuracy | 100.00% |
| 6 — Boosting (AdaBoost) | Accuracy | 100.00% |
| 6 — Bagging | Accuracy | 100.00% |
| 7 — KNN (K=14) | Accuracy | 95.56% |
| 8 — K-Means | Silhouette | 0.2287 |

---

## Important Limitations

### Assignment 1 — Salary Prediction Dataset Limitation

> **IMPORTANT**: A rigorous, programmatic search of the UCI ML Repository confirms
> that **no genuine continuous salary/income regression dataset** exists on UCI.
>
> - **UCI Census Income (Adult, ID: 20)** is a binary classification task (>50K vs ≤50K),
>   not a continuous salary dataset. Using it for regression would misrepresent both the
>   dataset and the assignment.
> - The popular "Salary_Data.csv" (YearsExperience vs. Salary) circulated online has no
>   verifiable UCI origin and would violate the requirement for a genuine UCI dataset.
>
> **Resolution**: UCI Auto MPG (ID: 9) is used as the academically defensible substitute.
> It is a genuine continuous regression dataset that cleanly demonstrates single-variable
> OLS regression (engine displacement → fuel economy). This limitation is fully documented
> in `assignment_1_simple_linear_regression/dataset_info.txt` and `results.txt`.

---

## Reproducibility

All scripts use `random_state=42` wherever supported, ensuring identical results
on any machine with the same package versions. Datasets are cached locally in
`datasets/` after first download.

---

## Academic Notes

- All datasets are from the UCI Machine Learning Repository.
- No synthetic or fabricated data is used.
- All metrics are generated from actual model execution.
- Code is written at B.Tech practical level — beginner-friendly with clear comments.
- Fixed seeds (`RANDOM_STATE=42`) ensure reproducible results.

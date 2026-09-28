"""
run_all.py — Master Execution Script for ML_Evaluation
Runs all 8 machine learning assignments sequentially,
captures PASS/FAIL status, and generates MASTER_RESULTS.txt
and MASTER_METRICS.csv.
"""

import os
import sys
import subprocess
import csv
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSIGNMENTS = [
    {
        "id": 1,
        "name": "Simple Linear Regression",
        "folder": "assignment_1_simple_linear_regression",
        "dataset": "UCI Auto MPG (ID: 9)",
        "algorithm": "LinearRegression (OLS)",
        "metrics_file": "metrics.csv",
        "results_file": "results.txt"
    },
    {
        "id": 2,
        "name": "Multiple Linear Regression",
        "folder": "assignment_2_multiple_linear_regression",
        "dataset": "UCI Real Estate Valuation (ID: 477)",
        "algorithm": "Multiple LinearRegression (OLS)",
        "metrics_file": "metrics.csv",
        "results_file": "results.txt"
    },
    {
        "id": 3,
        "name": "SVM for Email Classification",
        "folder": "assignment_3_svm_email",
        "dataset": "UCI Spambase (ID: 94)",
        "algorithm": "SVC (RBF Kernel)",
        "metrics_file": "metrics.csv",
        "results_file": "results.txt"
    },
    {
        "id": 4,
        "name": "Decision Tree for Spam Classification",
        "folder": "assignment_4_decision_tree_spam",
        "dataset": "UCI Spambase (ID: 94)",
        "algorithm": "DecisionTreeClassifier (Gini, max_depth=8)",
        "metrics_file": "metrics.csv",
        "results_file": "results.txt"
    },
    {
        "id": 5,
        "name": "Random Forest for Mushroom Classification",
        "folder": "assignment_5_random_forest_mushroom",
        "dataset": "UCI Mushroom (ID: 73)",
        "algorithm": "RandomForestClassifier (100 trees)",
        "metrics_file": "metrics.csv",
        "results_file": "results.txt"
    },
    {
        "id": 6,
        "name": "Boosting and Bagging for Wine Classification",
        "folder": "assignment_6_boosting_bagging_wine",
        "dataset": "UCI Wine (ID: 109)",
        "algorithm": "AdaBoostClassifier + BaggingClassifier",
        "metrics_file": "metrics.csv",
        "results_file": "results.txt"
    },
    {
        "id": 7,
        "name": "KNN for Iris Classification",
        "folder": "assignment_7_knn_iris",
        "dataset": "UCI Iris (ID: 53)",
        "algorithm": "KNeighborsClassifier (Best K via CV)",
        "metrics_file": "metrics.csv",
        "results_file": "results.txt"
    },
    {
        "id": 8,
        "name": "K-Means Clustering for Patient Grouping",
        "folder": "assignment_8_kmeans_patient_grouping",
        "dataset": "UCI Heart Disease (ID: 45)",
        "algorithm": "KMeans (k-means++, K=3)",
        "metrics_file": "metrics.csv",
        "results_file": "results.txt"
    }
]


def load_metrics(assignment_folder, metrics_file):
    """Load key metrics from a metrics.csv for summary reporting."""
    metrics_path = os.path.join(BASE_DIR, assignment_folder, metrics_file)
    if not os.path.exists(metrics_path):
        return {}
    metrics = {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics[row.get("Metric", "")] = row.get("Value", "")
    except Exception:
        pass
    return metrics


def run_all():
    print("=" * 75)
    print("ML EVALUATION — MASTER EXECUTION SCRIPT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    results_summary = []
    master_metrics_rows = []

    for assignment in ASSIGNMENTS:
        aid = assignment["id"]
        name = assignment["name"]
        folder = assignment["folder"]
        script_path = os.path.join(BASE_DIR, folder, "main.py")

        print(f"\n{'='*60}")
        print(f"  RUNNING ASSIGNMENT {aid}: {name}")
        print(f"{'='*60}")

        if not os.path.exists(script_path):
            print(f"  [ERROR] main.py not found: {script_path}")
            results_summary.append({
                "assignment": aid,
                "name": name,
                "status": "FAIL",
                "reason": "main.py not found",
                "dataset": assignment["dataset"],
                "algorithm": assignment["algorithm"]
            })
            continue

        start_time = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=os.path.join(BASE_DIR, folder),
                capture_output=False,
                timeout=300
            )
            elapsed = time.time() - start_time

            if proc.returncode == 0:
                print(f"\n  [PASS] Assignment {aid} completed in {elapsed:.1f}s")
                status = "PASS"
                reason = f"Completed in {elapsed:.1f}s"
            else:
                print(f"\n  [FAIL] Assignment {aid} returned exit code {proc.returncode}")
                status = "FAIL"
                reason = f"Exit code {proc.returncode}"

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"\n  [FAIL] Assignment {aid} timed out after {elapsed:.0f}s")
            status = "FAIL"
            reason = "Timeout"
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n  [FAIL] Assignment {aid} raised exception: {e}")
            status = "FAIL"
            reason = str(e)

        # Load metrics for summary
        metrics = load_metrics(folder, assignment["metrics_file"])

        results_summary.append({
            "assignment": aid,
            "name": name,
            "status": status,
            "reason": reason,
            "dataset": assignment["dataset"],
            "algorithm": assignment["algorithm"],
            "metrics": metrics
        })

        # Build master metrics row
        row = {
            "Assignment": aid,
            "Name": name,
            "Status": status,
            "Dataset": assignment["dataset"],
            "Algorithm": assignment["algorithm"]
        }
        row.update({f"Metric_{k}": v for k, v in metrics.items()})
        master_metrics_rows.append(row)

    # ── Print final summary ──────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("MASTER EXECUTION SUMMARY")
    print("=" * 75)
    pass_count = sum(1 for r in results_summary if r["status"] == "PASS")
    fail_count = len(results_summary) - pass_count
    print(f"Total: {len(results_summary)}  PASS: {pass_count}  FAIL: {fail_count}")
    print()
    for r in results_summary:
        icon = "[OK]" if r["status"] == "PASS" else "[XX]"
        print(f"  {icon} Assignment {r['assignment']:1d}: {r['name']:<45s} [{r['status']}]")

    # ── Write MASTER_RESULTS.txt ─────────────────────────────────────────
    master_results_path = os.path.join(BASE_DIR, "MASTER_RESULTS.txt")
    with open(master_results_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("ML EVALUATION — MASTER RESULTS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"TOTAL ASSIGNMENTS: {len(results_summary)}\n")
        f.write(f"PASSED:            {pass_count}\n")
        f.write(f"FAILED:            {fail_count}\n\n")
        f.write("-" * 75 + "\n\n")

        for r in results_summary:
            icon = "PASS" if r["status"] == "PASS" else "FAIL"
            f.write(f"Assignment {r['assignment']}: {r['name']}\n")
            f.write(f"  Status:     {icon}\n")
            f.write(f"  Dataset:    {r['dataset']}\n")
            f.write(f"  Algorithm:  {r['algorithm']}\n")
            f.write(f"  Detail:     {r['reason']}\n")

            metrics = r.get("metrics", {})
            if metrics:
                f.write("  Key Metrics:\n")
                for k, v in list(metrics.items())[:8]:
                    f.write(f"    {k}: {v}\n")

            result_file = os.path.join(BASE_DIR, r.get("name", ""), "results.txt")
            f.write(f"  Result Dir: {os.path.join(BASE_DIR, ASSIGNMENTS[r['assignment']-1]['folder'])}\n")
            f.write("\n")

        f.write("=" * 75 + "\n")
        f.write("END OF MASTER RESULTS REPORT\n")
        f.write("=" * 75 + "\n")

    print(f"\n[SAVED] MASTER_RESULTS.txt -> {master_results_path}")

    # ── Write MASTER_METRICS.csv ─────────────────────────────────────────
    if master_metrics_rows:
        # Collect all column names
        all_keys = []
        for row in master_metrics_rows:
            for k in row.keys():
                if k not in all_keys:
                    all_keys.append(k)

        master_csv_path = os.path.join(BASE_DIR, "MASTER_METRICS.csv")
        with open(master_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            for row in master_metrics_rows:
                writer.writerow(row)

        print(f"[SAVED] MASTER_METRICS.csv -> {master_csv_path}")

    print("\n" + "=" * 75)
    print(f"MASTER EXECUTION COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    return fail_count == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)

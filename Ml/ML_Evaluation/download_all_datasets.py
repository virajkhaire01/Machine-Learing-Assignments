"""
Utility script to fetch and save local copies of all UCI datasets for the 8 assignments.
Ensures local offline reproducibility while strictly adhering to genuine UCI sources.
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_datasets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, "datasets")
    
    datasets_info = [
        {
            "assignment": 1,
            "id": 9,
            "name": "Auto MPG",
            "filename": "auto_mpg.csv",
            "folder": "assignment_1"
        },
        {
            "assignment": 2,
            "id": 477,
            "name": "Real Estate Valuation",
            "filename": "real_estate_valuation.csv",
            "folder": "assignment_2"
        },
        {
            "assignment": 3,
            "id": 94,
            "name": "Spambase",
            "filename": "spambase.csv",
            "folder": "assignment_3"
        },
        {
            "assignment": 4,
            "id": 94,
            "name": "Spambase",
            "filename": "spambase.csv",
            "folder": "assignment_4"
        },
        {
            "assignment": 5,
            "id": 73,
            "name": "Mushroom",
            "filename": "mushroom.csv",
            "folder": "assignment_5"
        },
        {
            "assignment": 6,
            "id": 109,
            "name": "Wine",
            "filename": "wine.csv",
            "folder": "assignment_6"
        },
        {
            "assignment": 7,
            "id": 53,
            "name": "Iris",
            "filename": "iris.csv",
            "folder": "assignment_7"
        },
        {
            "assignment": 8,
            "id": 45,
            "name": "Heart Disease",
            "filename": "heart_disease.csv",
            "folder": "assignment_8"
        }
    ]
    
    print("=" * 60)
    print("FETCHING AND SAVING LOCAL COPIES OF UCI DATASETS")
    print("=" * 60)
    
    for item in datasets_info:
        target_path = os.path.join(datasets_dir, item["folder"], item["filename"])
        print(f"\n[Assignment {item['assignment']}] Fetching UCI ID {item['id']}: {item['name']}...")
        try:
            ds = fetch_ucirepo(id=item["id"])
            # Combine features and targets into one dataframe for local archival
            X = ds.data.features
            y = ds.data.targets
            
            if y is not None:
                df = pd.concat([X, y], axis=1)
            else:
                df = X.copy()
                
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            df.to_csv(target_path, index=False)
            print(f" -> Saved to: {target_path}")
            print(f" -> Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f" -> Error fetching ID {item['id']}: {e}")

    print("\nAll datasets processed successfully.")

if __name__ == "__main__":
    download_datasets()

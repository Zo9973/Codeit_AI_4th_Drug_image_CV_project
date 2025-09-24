import os
import glob
import pandas as pd

def analyze_latest_yolo_training_result(project_path='result/train'):
    result = sorted(glob.glob(os.path.join(project_path, '*')), key=os.path.getmtime, reverse=True)
    if not result:
        print("No training results found.")
        return {}

    latest_run = result[0]
    results_csv = os.path.join(latest_run, 'results.csv')
    best_model_path = os.path.join(latest_run, 'weights', 'best.pt')

    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        best_row = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
        print("\n🔍 Best Validation mAP:")
        print(best_row)
    else:
        print("No results.csv found.")

    return {
        "latest_run": latest_run,
        "best_model_path": best_model_path
    }

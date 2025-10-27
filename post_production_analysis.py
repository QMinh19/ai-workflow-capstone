import os
from datetime import timedelta, datetime
import json

from solution_guidance.model import model_predict, model_train
from src.models.monitoring import evaluate_predictions_with_truth

DATA_DIR = "cs-train"
LOG_DIR = os.environ.get("LOG_DIR","logs")

def simulate_days(start_date, end_date, country="all", retrain_every_n_days=7):
    dt = start_date
    counter = 0
    while dt < end_date:
        yyyy = str(dt.year)
        mm = str(dt.month).zfill(2)
        dd = str(dt.day).zfill(2)
        print(f"Predicting for {yyyy}-{mm}-{dd}")
        try:
            model_predict(country, yyyy, mm, dd)
        except Exception as e:
            print("Prediction failed:", e)
        counter += 1
        if counter % retrain_every_n_days == 0:
            print("Retraining model (periodic)...")
            model_train(DATA_DIR, test=True)
        dt += timedelta(days=1)

def compare_to_gold_standard(truth_csv):
    import pandas as pd
    truth = pd.read_csv(truth_csv)
    summary, plot_path = evaluate_predictions_with_truth(truth)
    print("Monitoring summary:", summary)
    print("Plot saved to", plot_path)

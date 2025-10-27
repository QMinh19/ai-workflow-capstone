import os
import pandas as pd
from solution_guidance.model import model_train
from solution_guidance.cslib import fetch_ts
from prophet import Prophet
import joblib

def train_all_models(data_dir='cs-train', test=False):
    print("=== Training Prophet baseline models ===")
    ts_data = fetch_ts(data_dir)
    if not os.path.exists("models"):
        os.makedirs("models")

    for country, df in ts_data.items():
        if test and country != "all":
            continue
        prophet_df = pd.DataFrame({'ds': df['date'], 'y': df['revenue']})
        model = Prophet()
        model.fit(prophet_df)
        model.save(f"models/prophet_{country}.json")
        print(f"Prophet model saved for {country}")

    print("=== Training RandomForest models ===")
    model_train(data_dir, test=test)
    print("All models trained and saved.")
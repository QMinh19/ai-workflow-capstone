import os
import pandas as pd
from solution_guidance.model import model_predict
from prophet import Prophet

def predict_with_all_models(country='all', year='2018', month='01', day='05'):
    rf_result = model_predict(country, year, month, day)

    prophet_path = f"models/prophet_{country}.json"
    if os.path.exists(prophet_path):
        model = Prophet()
        model = model.load(prophet_path)
        df_future = pd.DataFrame({'ds': [f"{year}-{month}-{day}"]})
        prophet_forecast = model.predict(df_future)
        prophet_pred = prophet_forecast['yhat'].iloc[0]
    else:
        prophet_pred = None

    return {
        "country": country,
        "date": f"{year}-{month}-{day}",
        "rf_prediction": rf_result["y_pred"][0],
        "prophet_prediction": prophet_pred
    }
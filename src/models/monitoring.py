import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR = os.environ.get("LOG_DIR", "logs")
REPORT_DIR = os.environ.get("REPORT_DIR", "reports")

def load_predict_log(path=None):
    if not path:
        path = os.path.join(LOG_DIR, "predict_log.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_prediction_trend(outdir=REPORT_DIR):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df = load_predict_log()
    df["pred"] = df["y_pred"].apply(lambda v: float(v[0]) if isinstance(v, list) else float(v))
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("ts")
    plt.figure(figsize=(10,4))
    plt.plot(df["ts"], df["pred"], marker="o", linestyle="-")
    plt.title("Predicted 30-day Revenue Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Predicted Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "predicted_revenue_trend.png"))
    return os.path.join(outdir, "predicted_revenue_trend.png")

def evaluate_predictions_with_truth(truth_df, outdir=REPORT_DIR):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    pred_df = load_predict_log()
    pred_df["pred"] = pred_df["y_pred"].apply(lambda v: float(v[0]) if isinstance(v, list) else float(v))
    pred_df["target_date"] = pd.to_datetime(pred_df["target_date"])
    truth_df = truth_df.copy()
    truth_df["date"] = pd.to_datetime(truth_df["date"])

    merged = pd.merge(pred_df, truth_df, left_on=["country","target_date"], right_on=["country","date"], how="inner")
    if merged.shape[0] == 0:
        raise ValueError("No matching predictions found in logs for given truth set")

    merged["abs_err"] = np.abs(merged["pred"] - merged["true_revenue"])
    merged["pct_err"] = merged["abs_err"] / (merged["true_revenue"].replace(0, np.nan))

    merged = merged.sort_values("timestamp")
    merged["rolling_mape"] = merged["pct_err"].rolling(window=7, min_periods=1).mean()

    plt.figure(figsize=(10,6))
    plt.plot(merged["target_date"], merged["true_revenue"], label="Actual")
    plt.plot(merged["target_date"], merged["pred"], label="Predicted")
    plt.legend()
    plt.title("Predicted vs Actual 30-day Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    out_path = os.path.join(outdir, "predicted_vs_actual.png")
    plt.savefig(out_path)

    summary = {
        "mae": float(merged["abs_err"].mean()),
        "median_abs_err": float(merged["abs_err"].median()),
        "mean_mape": float(merged["pct_err"].mean()),
        "latest_rolling_mape": float(merged["rolling_mape"].iloc[-1])
    }
    with open(os.path.join(outdir, "monitoring_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary, out_path

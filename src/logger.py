import os
import json
from datetime import datetime

LOG_DIR = os.environ.get("LOG_DIR", "logs")
TRAIN_LOG = os.path.join(LOG_DIR, "train_log.json")
PREDICT_LOG = os.path.join(LOG_DIR, "predict_log.json")

def _ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def _append_log(path, record):
    _ensure_log_dir()
    # create file if not exists
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f)
    with open(path, "r+", encoding="utf-8") as f:
        f.seek(0)
        data = json.load(f)
        data.append(record)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

def update_train_log(tag, date_range, metrics, runtime, model_version, note="", test=False):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "tag": tag,
        "date_range": date_range,
        "metrics": metrics,
        "runtime": runtime,
        "model_version": model_version,
        "note": note,
        "test": test
    }
    path = TRAIN_LOG if not test else os.path.join(LOG_DIR, "train_log_test.json")
    _append_log(path, record)
    return path

def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False):
    """
    Called by model.py after prediction.
    y_pred can be a scalar or list.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "country": country,
        "target_date": target_date,
        "y_pred": y_pred if isinstance(y_pred, (list,tuple)) else [float(y_pred)],
        "y_proba": y_proba,
        "runtime": runtime,
        "model_version": model_version,
        "test": test
    }
    path = PREDICT_LOG if not test else os.path.join(LOG_DIR, "predict_log_test.json")
    _append_log(path, record)
    return path
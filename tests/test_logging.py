import os
import json
from src.logger import update_predict_log, update_train_log
import tempfile

def test_update_predict_log(tmp_path):
    tmpdir = str(tmp_path)
    os.environ["LOG_DIR"] = tmpdir
    p = update_predict_log("all", [123.4], None, "2018-01-05", "00:00:01", 0.1, test=True)
    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert data[-1]["country"] == "all"

def test_update_train_log(tmp_path):
    tmpdir = str(tmp_path)
    os.environ["LOG_DIR"] = tmpdir
    p = update_train_log("all", ("2018-01-01","2018-01-31"), {"rmse": 100}, "00:01:00", 0.1, test=True)
    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)

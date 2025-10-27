from flask import Flask, request, jsonify
import os
import traceback

# preserve original names - import from uploaded guidance
from solution_guidance.model import model_train, model_predict

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train_endpoint():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        data_dir = payload.get("data_dir", "cs-train")
        test_flag = bool(payload.get("test", False))
        model_train(data_dir, test=test_flag)
        return jsonify({"status": "ok", "message": f"training started (test={test_flag})", "data_dir": data_dir}), 200
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": tb}), 500

@app.route("/predict", methods=["GET"])
def predict_endpoint():
    try:
        country = request.args.get("country", "all")
        year = request.args.get("year")
        month = request.args.get("month")
        day = request.args.get("day")
        test_flag = request.args.get("test", "false").lower() == "true"

        if not (year and month and day):
            return jsonify({"status": "error", "message": "year, month, day are required query params"}), 400

        result = model_predict(country, year, month, day, test=test_flag)
        y_pred = result.get("y_pred")
        y_proba = result.get("y_proba")
        return jsonify({
            "status": "ok",
            "country": country,
            "date": f"{year}-{month}-{day}",
            "y_pred": (list(y_pred) if hasattr(y_pred, "__iter__") else y_pred),
            "y_proba": (list(y_proba) if hasattr(y_proba, "__iter__") else y_proba)
        }), 200
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"status": "error", "message": str(e), "traceback": tb}), 500

@app.route("/logfile", methods=["GET"])
def logfile_endpoint():
    log_type = request.args.get("type", "predict")
    test_flag = request.args.get("test", "false").lower() == "true"
    log_dir = os.environ.get("LOG_DIR", "logs")
    if test_flag:
        file_map = {"train": os.path.join(log_dir, "train_log_test.json"),
                    "predict": os.path.join(log_dir, "predict_log_test.json")}
    else:
        file_map = {"train": os.path.join(log_dir, "train_log.json"),
                    "predict": os.path.join(log_dir, "predict_log.json")}
    path = file_map.get(log_type, file_map["predict"])
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": f"log file not found: {path}"}), 404
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return jsonify({"status": "ok", "path": path, "content": content}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
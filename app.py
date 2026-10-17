"""
STEP 3 - Flask website.

Folder layout expected:
    app.py
    predictor.py
    text_utils.py
    templates/index.html
    model/            <- produced by train_bert.py (config.json, model.safetensors, tokenizer files)

Run:  python app.py      then open http://127.0.0.1:5000
"""
import json
import os

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from predictor import FakeNewsPredictor

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
MAX_CHARS = 20000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")

app = Flask(__name__)
print(f"Loading model from {MODEL_DIR} ...")
predictor = FakeNewsPredictor(MODEL_DIR)
print("Model loaded. Labels:", predictor.id2label)


def load_json(name):
    """Read a results file from reports/ (returns None if it is missing)."""
    try:
        with open(os.path.join(REPORTS_DIR, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def analyze_page(**kw):
    kw.setdefault("text", "")
    kw.setdefault("result", None)
    kw.setdefault("error", None)
    return render_template("analyze.html", active="analyze", **kw)


@app.route("/", methods=["GET"])
def index():
    return analyze_page()


@app.route("/predict", methods=["POST"])
def predict():
    """Fallback for browsers without JavaScript (the page normally uses /api/predict)."""
    text = (request.form.get("news") or "").strip()[:MAX_CHARS]
    if not text:
        return analyze_page(error="Please paste some news text.")
    try:
        result = predictor.predict(text)
    except ValueError as e:
        return analyze_page(text=text, error=str(e))
    return analyze_page(text=text, result=result)


@app.route("/history")
def history():
    return render_template("history.html", active="history")


@app.route("/performance")
def performance():
    return render_template("performance.html", active="performance",
                           metrics=load_json("metrics.json"),
                           baselines=load_json("baselines.json"),
                           has_cm=os.path.exists(os.path.join(REPORTS_DIR, "confusion_matrix.png")))


@app.route("/how-it-works")
def how():
    return render_template("how.html", active="how")


@app.route("/about")
def about():
    return render_template("about.html", active="about")


@app.route("/reports/<path:name>")
def reports(name):
    if not name.lower().endswith(".png"):
        abort(404)
    return send_from_directory(REPORTS_DIR, name)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()[:MAX_CHARS]
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        return jsonify(predictor.predict(text))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

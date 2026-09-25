"""
app.py
------
Minimal Flask API around the trained diabetes-risk model.

Endpoints:
  GET  /api/health   -> service + model status
  POST /api/predict   -> { risk_score, risk_category, contributions, message }

The model is trained once (see train_model.py) and loaded at startup. If no
trained model is found on disk, one is trained automatically on first run.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
import train_model
from train_model import FEATURES, FEATURE_META, MODEL_DIR

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")




@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")



@app.after_request
def add_cors_headers(response):
    # Allow the static frontend (served from a different origin/port) to
    # call this API without needing the flask-cors package as a dependency.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/predict", methods=["OPTIONS"])
def predict_options():
    return "", 204

VALID_RANGES = {
    "pregnancies":      (0, 20),
    "glucose":          (40, 300),
    "blood_pressure":   (30, 180),
    "skin_thickness":   (0, 100),
    "insulin":          (0, 900),
    "bmi":              (10, 70),
    "diabetes_pedigree": (0.0, 3.0),
    "age":              (1, 120),
}

_model = None
_scaler = None
_meta = None


def load_or_train():
    """Load the trained model from disk, training it first if necessary."""
    global _model, _scaler, _meta
    model_path = MODEL_DIR / "model.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    meta_path = MODEL_DIR / "feature_meta.json"

    if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
        train_model.main()

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        _meta = json.load(f)


def categorize(risk_score: float) -> dict:
    if risk_score < 0.30:
        return {"level": "low", "label": "Lower risk"}
    if risk_score < 0.60:
        return {"level": "moderate", "label": "Moderate risk"}
    return {"level": "high", "label": "Higher risk"}


def build_explanation(raw_values: dict, scaled_row: np.ndarray) -> list:
    """
    Decompose the logistic regression prediction into a per-feature
    contribution using the model's own coefficients: for a logistic model,
    logit(p) = intercept + sum(coef_i * scaled_value_i), so coef_i *
    scaled_value_i is an exact, faithful measure of how much that feature
    pushed the prediction up or down. This is not an approximation layered
    on top of the model afterwards — it is the model's math itself, made
    readable.
    """
    coefs = _model.coef_[0]
    contributions = []
    for i, feature in enumerate(FEATURES):
        term = float(coefs[i] * scaled_row[0][i])
        contributions.append({
            "feature": feature,
            "label": FEATURE_META[feature]["label"],
            "unit": FEATURE_META[feature]["unit"],
            "value": raw_values[feature],
            "contribution": term,
            "direction": "increases" if term > 0 else "decreases",
        })

    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    max_abs = max(abs(c["contribution"]) for c in contributions) or 1.0
    for c in contributions:
        magnitude = abs(c["contribution"]) / max_abs
        if magnitude > 0.66:
            c["impact"] = "high"
        elif magnitude > 0.33:
            c["impact"] = "medium"
        else:
            c["impact"] = "low"

    return contributions


def plain_language_summary(category: dict, contributions: list) -> str:
    top = [c for c in contributions if c["direction"] == "increases"][:2]
    if not top:
        return (
            "None of the values entered are currently pushing your estimated "
            "risk up in a meaningful way."
        )
    parts = [f"{c['label'].lower()} ({c['value']}{(' ' + c['unit']) if c['unit'] else ''})" for c in top]
    joined = " and ".join(parts)
    return f"Your {joined} are the biggest factors raising your estimated risk."


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "model_metrics": _meta.get("metrics") if _meta else None,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    values = {}
    for f in FEATURES:
        try:
            values[f] = float(data[f])
        except (TypeError, ValueError):
            return jsonify({"error": f"Field '{f}' must be a number."}), 400
        lo, hi = VALID_RANGES[f]
        if not (lo <= values[f] <= hi):
            return jsonify({
                "error": f"Field '{f}' should be between {lo} and {hi}."
            }), 400

    row = np.array([[values[f] for f in FEATURES]])
    scaled_row = _scaler.transform(row)

    risk_score = float(_model.predict_proba(scaled_row)[0][1])
    category = categorize(risk_score)
    contributions = build_explanation(values, scaled_row)
    summary = plain_language_summary(category, contributions)

    return jsonify({
        "risk_score": round(risk_score, 4),
        "risk_percent": round(risk_score * 100, 1),
        "risk_category": category,
        "summary": summary,
        "contributions": contributions,
        "disclaimer": (
            "This tool gives an educational estimate based on a statistical "
            "model. It is not a medical diagnosis. Please talk to a "
            "healthcare professional about your personal risk."
        ),
    })


load_or_train()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

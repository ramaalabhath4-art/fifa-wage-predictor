# =============================================================
# app.py  –  FIFA Wage Predictor  |  Flask Backend
# =============================================================
# Run:   python app.py
# Open:  http://localhost:5000
# =============================================================

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(ROOT_DIR, "models", "best_model.joblib")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "features.json")

# ── Load model once at startup ─────────────────────────────────
print("Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n❌  Model not found at: {MODEL_PATH}\n"
        "    Please run:  python src/train.py  first!\n"
    )
model    = joblib.load(MODEL_PATH)
metadata = json.load(open(METADATA_PATH, encoding="utf-8"))
print(f"✅  Model loaded: {metadata['best_model_name']}  |  R² = {metadata['best_r2']:.4f}")

# ── Wage tier definitions ──────────────────────────────────────
WAGE_TIERS = [
    (0,    10,   "Low",        "🟤", "#a0522d"),
    (10,   30,   "Below Avg",  "🔵", "#4682b4"),
    (30,   80,   "Average",    "🟢", "#2e8b57"),
    (80,   200,  "Above Avg",  "🟡", "#daa520"),
    (200,  500,  "High",       "🟠", "#ff8c00"),
    (500,  9999, "Elite",      "🔴", "#dc143c"),
]

WAGE_TABLE = [
    (91,99,380),(88,91,215),(85,88,148),(82,85,72),
    (79,82,42),(76,79,26),(73,76,17),(70,73,9),
    (65,70,3),(60,65,2),(0,60,1),
]

def get_tier(wage):
    for lo, hi, label, icon, color in WAGE_TIERS:
        if lo <= wage < hi:
            return label, icon, color
    return "Elite", "🔴", "#dc143c"

def impute_wage(overall: int) -> int:
    for lo, hi, med in WAGE_TABLE:
        if lo <= overall < hi:
            return med
    return 1

def tier_probabilities(pred_wage: float, sigma: float = 0.18) -> dict:
    rng      = np.random.default_rng(42)
    pred_log = np.log1p(pred_wage)
    samples  = np.expm1(rng.normal(pred_log, sigma, size=20_000))
    probs    = {}
    for lo, hi, label, _, _ in WAGE_TIERS:
        count        = int(np.sum((samples >= lo) & (samples < hi)))
        probs[label] = round(count / len(samples) * 100, 1)
    return probs


# ── Routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        model_name = metadata["best_model_name"],
        r2         = metadata["best_r2"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    wage = int(data.get("Wage", 0))
    if wage == 0:
        wage = impute_wage(int(data.get("Overall", 75)))

    player = {
        "Overall":        int(data["Overall"]),
        "Potential":      int(data["Potential"]),
        "Value":          float(data["Value"]),
        "Release Clause": float(data["Release Clause"]),
        "Age":            int(data["Age"]),
        "Wage":           wage,
        "Position":       str(data["Position"]),
        "Preferred Foot": str(data["Preferred Foot"]),
        "Club":           str(data["Club"]),
    }

    df_input  = pd.DataFrame([player])
    pred_log  = float(model.predict(df_input)[0])
    pred_wage = float(np.expm1(pred_log))
    label, icon, color = get_tier(pred_wage)

    return jsonify({
        "predicted_wage": round(pred_wage, 2),
        "predicted_log":  round(pred_log, 4),
        "tier_label":     label,
        "tier_icon":      icon,
        "tier_color":     color,
        "tier_probs":     tier_probabilities(pred_wage),
        "wage_used":      wage,
        "model_name":     metadata["best_model_name"],
        "r2":             metadata["best_r2"],
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data     = request.get_json()
    players  = data.get("players", [])
    features = metadata["selected_features"]
    results  = []

    for p in players:
        wage = int(float(p.get("Wage", 0)))
        if wage == 0:
            wage = impute_wage(int(float(p.get("Overall", 75))))
        p["Wage"] = wage

        row = {f: p.get(f) for f in features}
        try:
            pred_log  = float(model.predict(pd.DataFrame([row]))[0])
            pred_wage = float(np.expm1(pred_log))
            label, icon, color = get_tier(pred_wage)
            results.append({
                **p,
                "Predicted_Wage": round(pred_wage, 2),
                "Wage_Tier":      label,
                "Tier_Color":     color,
            })
        except Exception as e:
            results.append({**p, "error": str(e)})

    return jsonify({"results": results})


if __name__ == "__main__":
    print("\n⚽   FIFA Wage Predictor is running!")
    print("     Open your browser:  http://localhost:5000\n")
    app.run(debug=True, port=5000)

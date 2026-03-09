# =============================================================
# predict.py  –  FIFA Wage Predictor  |  Make Predictions
# =============================================================
# Run:  python src/predict.py
# =============================================================

import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# determine workspace root (three levels above this file) so data/ and models/
# are always referenced consistently regardless of working directory.
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH    = os.path.join(ROOT_DIR, "models", "best_model.joblib")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "features.json")


# =============================================================
# LOAD MODEL + METADATA
# =============================================================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            "Run:  python src/train.py   first!"
        )
    model    = joblib.load(MODEL_PATH)
    metadata = json.load(open(METADATA_PATH, encoding="utf-8"))
    print(f"✅ Loaded model  : {metadata['best_model_name']}")
    print(f"   R² score      : {metadata['best_r2']:.4f}")
    print(f"   Features used : {metadata['selected_features']}\n")
    return model, metadata


# =============================================================
# PREDICT SINGLE PLAYER
# =============================================================
def predict_player(model, player: dict) -> dict:
    """
    Predict weekly wage for one player.

    Parameters
    ----------
    player : dict with keys matching selected_features
        Example:
            {
                "Overall":       85,
                "Potential":     88,
                "Value":         45000,
                "Release Clause":90000,
                "Age":           24,
                "Wage":          0,        # unknown → set to 0
                "Position":     "ST",
                "Preferred Foot":"Right",
                "Club":         "Arsenal"
            }

    Returns
    -------
    dict with predicted_wage_log and predicted_wage_weekly
    """
    df_input    = pd.DataFrame([player])
    pred_log    = model.predict(df_input)[0]
    pred_wage   = np.expm1(pred_log)          # reverse log1p

    return {
        "predicted_log_wage":    round(pred_log,  4),
        "predicted_weekly_wage": round(pred_wage, 2)
    }


# =============================================================
# PREDICT BATCH FROM CSV
# =============================================================
def predict_batch(model, metadata, csv_path: str, output_path: str = None):
    """
    Predict wages for all players in a CSV file.
    CSV must contain the required feature columns.
    """
    df      = pd.read_csv(csv_path)
    features = metadata["selected_features"]

    # Check all features are present
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    X           = df[features]
    preds_log   = model.predict(X)
    preds_wage  = np.expm1(preds_log)

    df["Predicted_Wage_Log"]    = preds_log.round(4)
    df["Predicted_Weekly_Wage"] = preds_wage.round(2)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✅ Batch predictions saved → {output_path}")
    else:
        print(df[["Predicted_Wage_Log", "Predicted_Weekly_Wage"]].head(10))

    return df


# =============================================================
# INTERACTIVE MODE
# =============================================================
def interactive_mode(model):
    print("=" * 50)
    print("  FIFA Wage Predictor – Interactive Mode")
    print("=" * 50)
    print("Enter player details (press Enter to use default)\n")

    def ask(prompt, default, cast=str):
        val = input(f"  {prompt} [{default}]: ").strip()
        return cast(val) if val else cast(default)

    player = {
        "Overall":       ask("Overall rating (46–94)",     85,       int),
        "Potential":     ask("Potential (48–95)",           88,       int),
        "Value":         ask("Market value (€K)",           45000,    float),
        "Release Clause":ask("Release clause (€K)",         90000,    float),
        "Age":           ask("Age",                         24,       int),
        "Wage":          0,                                           # unknown
        "Position":      ask("Position (ST/GK/CM/...)",    "ST",     str),
        "Preferred Foot":ask("Preferred foot (Right/Left)", "Right",  str),
        "Club":          ask("Club",                        "Arsenal", str),
    }

    result = predict_player(model, player)
    print("\n" + "─" * 40)
    print(f"  📊 Predicted weekly wage : €{result['predicted_weekly_wage']:,.0f}K")
    print(f"  📉 Log-wage value        : {result['predicted_log_wage']}")
    print("─" * 40)


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    model, metadata = load_model()

    # ── Demo: hardcoded example players ──────────────────────────────────────
    print("─" * 50)
    print("  Demo predictions")
    print("─" * 50)

    demo_players = [
        {
            "Overall": 94, "Potential": 94, "Value": 110500,
            "Release Clause": 226500, "Age": 31,  "Wage": 0,
            "Position": "RF",  "Preferred Foot": "Left",  "Club": "FC Barcelona"
        },
        {
            "Overall": 80, "Potential": 85, "Value": 30000,
            "Release Clause": 60000,  "Age": 22,  "Wage": 0,
            "Position": "CM",  "Preferred Foot": "Right", "Club": "Arsenal"
        },
        {
            "Overall": 65, "Potential": 72, "Value": 500,
            "Release Clause": 1000,   "Age": 19,  "Wage": 0,
            "Position": "GK",  "Preferred Foot": "Right", "Club": "Leicester City"
        },
    ]

    for i, p in enumerate(demo_players, 1):
        result = predict_player(model, p)
        print(f"\n  Player {i}: {p['Position']} | Overall {p['Overall']} | Age {p['Age']}")
        print(f"    → Predicted wage : €{result['predicted_weekly_wage']:,.0f}K / week")

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("\n")
    run_interactive = input("Run interactive mode? (y/n): ").strip().lower()
    if run_interactive == "y":
        interactive_mode(model)

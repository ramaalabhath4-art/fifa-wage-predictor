# =============================================================
# evaluate.py  –  FIFA Wage Predictor  |  Model Evaluation
# =============================================================
# Run:  python src/evaluate.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
# determine workspace root (three levels above this script)
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH    = os.path.join(ROOT_DIR, "models", "best_model.joblib")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "features.json")
DATA_PATH     = os.path.join(ROOT_DIR, "data",   "fifa_eda.csv")
MODEL_DIR     = os.path.join(ROOT_DIR, "models")

# =============================================================
# LOAD
# =============================================================
print("Loading model and data ...")

model    = joblib.load(MODEL_PATH)
metadata = json.load(open(METADATA_PATH, encoding="utf-8"))

df       = pd.read_csv(DATA_PATH)
df_clean = df.copy()
df_clean = df_clean.dropna(subset=["Wage"])
df_clean = df_clean[df_clean["Wage"] > 0]
df_clean["Wage_log"] = np.log1p(df_clean["Wage"])

for col in df_clean.select_dtypes(include="object").columns:
    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
for col in df_clean.select_dtypes(include=["int64", "float64"]).columns:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

features = metadata["selected_features"]
X        = df_clean[features]
y_true   = np.log1p(df_clean["Wage"])

# =============================================================
# PREDICT & METRICS
# =============================================================
print("\nRunning evaluation ...")

y_pred_log  = model.predict(X)
y_pred_wage = np.expm1(y_pred_log)
y_true_wage = np.expm1(y_true)

mae_log  = mean_absolute_error(y_true,      y_pred_log)
mae_wage = mean_absolute_error(y_true_wage, y_pred_wage)
rmse     = np.sqrt(mean_squared_error(y_true, y_pred_log))
r2       = r2_score(y_true, y_pred_log)

print("\n" + "=" * 45)
print("  Evaluation Results")
print("=" * 45)
print(f"  R²   (log scale)   : {r2:.4f}")
print(f"  MAE  (log scale)   : {mae_log:.4f}")
print(f"  RMSE (log scale)   : {rmse:.4f}")
print(f"  MAE  (actual wage) : €{mae_wage:,.1f}K / week")
print("=" * 45)

# =============================================================
# PLOTS
# =============================================================

# 1. Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred_log, alpha=0.3, color="steelblue", s=10)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()], "r--", linewidth=2, label="Perfect fit")
plt.xlabel("Actual log(Wage)")
plt.ylabel("Predicted log(Wage)")
plt.title(f"Actual vs Predicted  |  R² = {r2:.4f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eval_actual_vs_predicted.png"), dpi=120)
plt.close()
print("\n✅ Plot saved: eval_actual_vs_predicted.png")

# 2. Residuals
residuals = y_true - y_pred_log

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_log, residuals, alpha=0.3, color="salmon", s=10)
plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
plt.xlabel("Predicted log(Wage)")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eval_residuals.png"), dpi=120)
plt.close()
print("✅ Plot saved: eval_residuals.png")

# 3. Residual distribution
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=60, color="steelblue", edgecolor="white")
plt.axvline(0, color="red", linewidth=1.5, linestyle="--")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Residual Distribution")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eval_residual_dist.png"), dpi=120)
plt.close()
print("✅ Plot saved: eval_residual_dist.png")

# 4. Predicted wage distribution
plt.figure(figsize=(10, 5))
plt.hist(y_true_wage,  bins=60, alpha=0.6, color="steelblue", label="Actual Wage")
plt.hist(y_pred_wage,  bins=60, alpha=0.6, color="salmon",    label="Predicted Wage")
plt.xlabel("Weekly Wage (€K)")
plt.ylabel("Count")
plt.title("Actual vs Predicted Wage Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eval_wage_distribution.png"), dpi=120)
plt.close()
print("✅ Plot saved: eval_wage_distribution.png")

print("\nEvaluation complete! Check models/ folder for all plots.")

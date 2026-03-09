# =============================================================
# train.py  –  FIFA Wage Predictor  |  Full Training Pipeline
# =============================================================
# Run:  python src/train.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection      import train_test_split, cross_val_score
from sklearn.impute               import SimpleImputer
from sklearn.preprocessing        import StandardScaler, OneHotEncoder
from sklearn.linear_model         import LinearRegression
from sklearn.ensemble             import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose              import ColumnTransformer
from sklearn.pipeline             import Pipeline
from sklearn.metrics              import mean_absolute_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
# compute workspace root (three levels above this file) so scripts can run
# whether executed from project root or inner directory.
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = "data/fifa_eda.csv"
MODEL_DIR  = os.path.join(ROOT_DIR, "models")
# ensure output structure exists
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================
# 1) LOAD DATA
# =============================================================
print("=" * 55)
print("  FIFA Wage Predictor — Training Pipeline")
print("=" * 55)

print("\n[1] Loading data ...")
df     = pd.read_csv(DATA_PATH)
df_raw = df.copy()

print(f"    Shape        : {df.shape}")
print(f"    Missing vals :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# =============================================================
# 2) EDA PLOTS  (saved to models/ folder)
# =============================================================
print("\n[2] Generating EDA plots ...")

# Wage distribution
plt.figure(figsize=(10, 5))
plt.hist(df["Wage"], bins=50, color="steelblue", edgecolor="white")
plt.title("Wage Distribution (Before Cleaning)")
plt.xlabel("Weekly Wage")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "plot_wage_dist.png"), dpi=100)
plt.close()

# Overall vs Wage
plt.figure(figsize=(10, 5))
plt.scatter(df["Overall"], df["Wage"], alpha=0.4, color="steelblue")
plt.xlabel("Overall Rating")
plt.ylabel("Wage")
plt.title("Overall vs Wage")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "plot_overall_vs_wage.png"), dpi=100)
plt.close()

# Position bar chart
position_wage = (
    df.groupby("Position")["Wage"].mean()
    .sort_values(ascending=False).head(10)
)
plt.figure(figsize=(10, 6))
position_wage.plot(kind="bar", color="steelblue")
plt.title("Average Weekly Wage by Position (Top 10)")
plt.xlabel("Position")
plt.ylabel("Average Wage")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "plot_position_wage.png"), dpi=100)
plt.close()

print("    Plots saved to models/")

# =============================================================
# 3) DATA CLEANING
# =============================================================
print("\n[3] Cleaning data ...")

df_clean = df_raw.copy()
before   = len(df_clean)

df_clean = df_clean.dropna(subset=["Wage"])
df_clean = df_clean[df_clean["Wage"] > 0]

print(f"    Rows before : {before}")
print(f"    Rows after  : {len(df_clean)}")

# Log-transform target
df_clean["Wage_log"] = np.log1p(df_clean["Wage"])

# Fill missing values
for col in df_clean.select_dtypes(include="object").columns:
    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

for col in df_clean.select_dtypes(include=["int64", "float64"]).columns:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

print(f"    Remaining NaN: {df_clean.isnull().sum().sum()}")

# Before / After / Log plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(df_raw["Wage"],        bins=50, color="salmon",    edgecolor="white")
axes[0].set_title("Before Cleaning")
axes[1].hist(df_clean["Wage"],      bins=50, color="steelblue", edgecolor="white")
axes[1].set_title("After Cleaning")
axes[2].hist(df_clean["Wage_log"],  bins=50, color="seagreen",  edgecolor="white")
axes[2].set_title("Log-Transformed Wage")
plt.suptitle("Wage Distribution: Before / After / Log", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "plot_wage_cleaning.png"), dpi=100)
plt.close()

# =============================================================
# 4) FEATURE ENGINEERING
# =============================================================
print("\n[4] Feature engineering ...")

selected_features    = ["Overall", "Potential", "Value", "Release Clause",
                        "Age", "Wage", "Position", "Preferred Foot", "Club"]
numeric_features     = ["Overall", "Potential", "Value", "Release Clause", "Age", "Wage"]
categorical_features = ["Position", "Preferred Foot", "Club"]

X = df_clean[selected_features]
y = np.log1p(df_clean["Wage"])

print(f"    X shape : {X.shape}")
print(f"    y shape : {y.shape}")

# =============================================================
# 5) PREPROCESSING PIPELINE
# =============================================================
print("\n[5] Building preprocessor ...")

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# =============================================================
# 6) TRAIN / TEST SPLIT
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[6] Split → Train: {len(X_train)}  |  Test: {len(X_test)}")

# =============================================================
# 7) TRAIN MODELS
# =============================================================
print("\n[7] Training models ...")

def build_pipeline(model):
    return Pipeline([
        ("preprocessing", preprocessor),
        ("model",         model)
    ])

# Linear Regression
print("    → Linear Regression ...")
pipeline_lr = build_pipeline(LinearRegression())
pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr  = r2_score(y_test, y_pred_lr)
print(f"       MAE: {mae_lr:.4f}  |  R²: {r2_lr:.4f}")

# Random Forest
print("    → Random Forest ...")
pipeline_rf = build_pipeline(
    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
)
pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf  = r2_score(y_test, y_pred_rf)
print(f"       MAE: {mae_rf:.4f}  |  R²: {r2_rf:.4f}")

# HistGradientBoosting
print("    → HistGradientBoosting (CV 5-fold) ...")
pipeline_hgb = build_pipeline(
    HistGradientBoostingRegressor(max_iter=500, max_depth=20, random_state=42)
)
cv_r2 = cross_val_score(pipeline_hgb, X, y, cv=5, scoring="r2")
print(f"       CV R² mean: {cv_r2.mean():.4f}  |  std: {cv_r2.std():.4f}")

pipeline_hgb.fit(X_train, y_train)
y_pred_hgb = pipeline_hgb.predict(X_test)
mae_hgb = mean_absolute_error(y_test, y_pred_hgb)
r2_hgb  = r2_score(y_test, y_pred_hgb)
print(f"       MAE: {mae_hgb:.4f}  |  R²: {r2_hgb:.4f}")

# =============================================================
# 8) COMPARE & SELECT BEST
# =============================================================
print("\n[8] Model comparison ...")

results_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "HGB"],
    "MAE":   [mae_lr,  mae_rf,  mae_hgb],
    "R2":    [r2_lr,   r2_rf,   r2_hgb]
}).sort_values("R2", ascending=False).reset_index(drop=True)

print(results_df.to_string(index=False))

cv_scores = {
    "Linear Regression": r2_lr,
    "Random Forest":     r2_rf,
    "HGB":               r2_hgb
}
trained_pipelines = {
    "Linear Regression": pipeline_lr,
    "Random Forest":     pipeline_rf,
    "HGB":               pipeline_hgb
}

best_model_name = max(cv_scores, key=cv_scores.get)
best_estimator  = trained_pipelines[best_model_name]
print(f"\n    🏆 Best model : {best_model_name}  (R² = {cv_scores[best_model_name]:.4f})")

# Actual vs Predicted plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_hgb, alpha=0.4, color="steelblue")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", linewidth=2)
plt.xlabel("Actual Log(Wage)")
plt.ylabel("Predicted Log(Wage)")
plt.title("HGB – Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "plot_actual_vs_predicted.png"), dpi=100)
plt.close()

# Residuals plot
residuals = y_test - y_pred_hgb
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_hgb, residuals, alpha=0.4, color="salmon")
plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
plt.xlabel("Predicted Log(Wage)")
plt.ylabel("Residuals")
plt.title("HGB – Residuals Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "plot_residuals.png"), dpi=100)
plt.close()

# =============================================================
# 9) SAVE BEST MODEL
# =============================================================
print("\n[9] Saving model ...")

model_path    = os.path.join(MODEL_DIR, "best_model.joblib")
metadata_path = os.path.join(MODEL_DIR, "features.json")
results_path  = os.path.join(MODEL_DIR, "gridsearch_results.csv")

metadata = {
    "best_model_name":      best_model_name,
    "best_r2":              cv_scores[best_model_name],
    "all_scores":           cv_scores,
    "selected_features":    selected_features,
    "numeric_features":     numeric_features,
    "categorical_features": categorical_features,
    "target":               "Wage_log"
}

try:
    joblib.dump(best_estimator, model_path)
    print(f"    ✅ Model    → {model_path}")
except Exception as e:
    print(f"    ❌ Model save failed : {e}")

try:
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    print(f"    ✅ Metadata → {metadata_path}")
except Exception as e:
    print(f"    ❌ JSON save failed  : {e}")

try:
    results_df.to_csv(results_path, index=False)
    print(f"    ✅ Results  → {results_path}")
except Exception as e:
    print(f"    ❌ CSV save failed   : {e}")

print("\n" + "=" * 55)
print("  Training complete!")
print("=" * 55)

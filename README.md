# ⚽ FIFA Wage Predictor — Full Project

## 📁 Project Structure

```
fifa_wage_predictor/
│
├── data/
│   └── fifa_eda.csv          ← dataset
│
├── models/                   ← created automatically after training
│   ├── best_model.joblib
│   └── features.json
│
├── src/
│   ├── train.py              ← train the model
│   ├── predict.py            ← test predictions in terminal
│   └── evaluate.py           ← evaluate model performance
│
├── templates/
│   └── index.html            ← web interface (HTML + CSS)
│
├── app.py                    ← Flask web server
├── requirements.txt
├── README.md
└── streamlit_app.py
```

---

## ▶️ Step-by-Step Setup

### Step 1 — Open terminal in your project folder
```bash
cd D:\fifi\fifa_wage_predictor
```

### Step 2 — Activate your virtual environment
```bash
.venv\Scripts\activate
```

### Step 3 — Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the model (creates models/ folder)
```bash
python src/train.py
```
This creates:
- `models/best_model.joblib`
- `models/features.json`
- Several plots in `models/`

### Step 5 — Run the Flask web app
```bash
python app.py
```

### Step 6 — Open in browser
```
http://localhost:5000
```

---

## 🌐 How the Flask App Works

```
Browser (HTML + CSS)
      ↓  fill form → click Predict
Flask /predict endpoint
      ↓  loads best_model.joblib
Real ML Model (Random Forest, R²=0.9999)
      ↓  returns predicted wage + tier probabilities
Browser shows result with animated bars
```

---

## 📊 Model Results

| Model             | R²     |
|-------------------|--------|
| Random Forest     | ~0.9999|
| HGB               | ~0.9998|
| Linear Regression | ~0.91  |

---

## 🎯 API Endpoints

| Endpoint          | Method | Description              |
|-------------------|--------|--------------------------|
| `/`               | GET    | Web interface            |
| `/predict`        | POST   | Single player prediction |
| `/predict_batch`  | POST   | Batch CSV prediction     |

# 🎓 Student Performance Prediction & Early Warning System

A full-stack ML system that predicts student academic risk (Low / Medium / High)
and delivers early warning alerts, explainability, and optional email notifications.

---

## 📁 Project Structure

```
student_performance/
├── data/
│   ├── generate_dataset.py     ← Synthetic dataset generator
│   └── student_data.csv        ← Auto-generated on first run
├── model/
│   ├── train_model.py          ← ML training script
│   ├── model.pkl               ← Saved best model (after training)
│   ├── scaler.pkl              ← Feature scaler
│   ├── label_encoder.pkl       ← Label encoder
│   └── rf_importances.pkl      ← Feature importances for XAI
├── backend/
│   └── main.py                 ← FastAPI backend
├── frontend/
│   └── app.py                  ← Streamlit dashboard
├── logs/
│   └── predictions.csv         ← Auto-created prediction log
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# 1. Clone / download the project
cd student_performance

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1 — Generate Dataset & Train Model

```bash
# Generate synthetic data (creates data/student_data.csv)
python data/generate_dataset.py

# Train models and save artifacts to model/
python model/train_model.py
```

Expected output:
```
── Random Forest ──   Accuracy : 0.97xx   F1 : 0.97xx
── XGBoost ──         Accuracy : 0.98xx   F1 : 0.98xx
✅ Best model → XGBoost
Artifacts saved in model/
```

---

### Step 2 — Start the FastAPI Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

---

### Step 3 — Launch the Streamlit Dashboard

Open a new terminal:

```bash
cd frontend
streamlit run app.py
```

Dashboard opens at: http://localhost:8501

---

## 📡 API Endpoints

| Method | Endpoint       | Description                        |
|--------|----------------|------------------------------------|
| GET    | /              | Health check                       |
| POST   | /predict       | Predict risk for one student       |
| POST   | /upload-data   | Upload CSV for retraining          |
| POST   | /retrain       | Merge data & retrain model         |
| GET    | /logs          | Fetch prediction history           |
| GET    | /stats         | Summary stats                      |

### Example `/predict` Request

```json
{
  "student_name":   "Alice Smith",
  "email":          "alice@college.edu",
  "attendance":     52.0,
  "prev_gpa":       4.2,
  "internal_marks": 38.0,
  "backlogs":       3
}
```

---

## 📧 Email Alert Configuration

Set environment variables before starting the backend:

```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER=your.email@gmail.com
export SMTP_PASS=your_app_password    # Gmail App Password, NOT your login password
```

Emails are triggered automatically when `risk_level == "High"` and an email address is provided.

---

## 🔄 Retraining Workflow

1. Go to **Upload & Retrain** page in the dashboard
2. Upload a CSV with columns: `attendance, prev_gpa, internal_marks, backlogs, risk_level`
3. Click **Upload File**, then **Retrain Model**
4. The backend merges old + new data, retrains, and hot-swaps the model

---

## 🧠 Model Details

| Model         | Features Used                              |
|---------------|--------------------------------------------|
| Random Forest | attendance, prev_gpa, internal_marks, backlogs |
| XGBoost       | Same — best model is auto-selected by F1   |

### Risk Label Rules

| Risk   | Condition                                          |
|--------|----------------------------------------------------|
| High   | attendance < 55 OR GPA < 4.5 OR marks < 35 OR backlogs ≥ 4 |
| Low    | attendance ≥ 75 AND GPA ≥ 7 AND marks ≥ 60 AND backlogs = 0 |
| Medium | Everything else                                    |

---

## 🛠️ Troubleshooting

| Issue                         | Fix                                          |
|-------------------------------|----------------------------------------------|
| `model.pkl not found`         | Run `python model/train_model.py` first      |
| Backend 503 error             | Train the model — artifacts are missing      |
| Cannot connect to backend     | Ensure FastAPI is running on port 8000       |
| Email not sending             | Set SMTP_* environment variables correctly   |
| XGBoost install error (Mac M1)| `pip install xgboost --no-binary :all:`      |

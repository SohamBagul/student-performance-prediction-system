"""
main.py — FastAPI Backend
=========================
Endpoints:
  POST /predict        → Risk prediction + alerts + explanation
  POST /upload-data    → Accept CSV for retraining
  POST /retrain        → Merge datasets & retrain model
  GET  /logs           → Return prediction history
"""

import os, csv, pickle, logging, smtplib, subprocess, sys
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODEL_DIR   = os.path.join(BASE_DIR, "..", "model")
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
LOGS_DIR    = os.path.join(BASE_DIR, "..", "logs")
LOG_FILE    = os.path.join(LOGS_DIR, "predictions.csv")
UPLOAD_FILE = os.path.join(DATA_DIR, "uploaded_data.csv")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Load model artifacts ──────────────────────────────────────────────────────
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "model.pkl"),         "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),        "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "rf_importances.pkl"), "rb") as f:
        importances = pickle.load(f)
    return model, scaler, le, importances

try:
    model, scaler, le, importances = load_artifacts()
    logger.info("Model artifacts loaded successfully.")
except Exception as e:
    logger.error(f"Could not load model: {e}")
    model = scaler = le = importances = None

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict academic risk and generate early warnings.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

FEATURES = ["attendance", "prev_gpa", "internal_marks", "backlogs"]

# ── Schemas ───────────────────────────────────────────────────────────────────
class StudentInput(BaseModel):
    attendance:     float = Field(..., ge=0, le=100, description="Attendance %")
    prev_gpa:       float = Field(..., ge=0, le=10,  description="Previous GPA (0–10)")
    internal_marks: float = Field(..., ge=0, le=100, description="Internal marks (0–100)")
    backlogs:       int   = Field(..., ge=0,          description="Number of backlogs")
    student_name:   Optional[str] = "Unknown"
    email:          Optional[str] = None

class PredictionResponse(BaseModel):
    student_name:  str
    risk_level:    str
    confidence:    float
    alerts:        list[str]
    explanation:   str
    top_factors:   list[str]

# ── Helper: Rule-based alerts ─────────────────────────────────────────────────
def generate_alerts(data: StudentInput) -> list[str]:
    alerts = []
    if data.attendance < 60:
        alerts.append("⚠️ Low Attendance Warning: Attendance below 60%")
    if data.prev_gpa < 5.0:
        alerts.append("📉 Performance Decline Alert: GPA below 5.0")
    if data.backlogs > 2:
        alerts.append("🚨 High Academic Risk: More than 2 active backlogs")
    if data.internal_marks < 40:
        alerts.append("📋 Low Internal Marks: Marks below 40%")
    return alerts

# ── Helper: Explanation from importances ──────────────────────────────────────
def generate_explanation(data: StudentInput, risk: str, imp: dict) -> tuple[str, list[str]]:
    values = {
        "attendance":     data.attendance,
        "prev_gpa":       data.prev_gpa,
        "internal_marks": data.internal_marks,
        "backlogs":       data.backlogs,
    }
    # rank features by importance
    ranked = sorted(imp.items(), key=lambda x: -x[1])
    top_factors = []
    reasons = []

    readable = {
        "attendance":     f"attendance ({data.attendance:.1f}%)",
        "prev_gpa":       f"GPA ({data.prev_gpa:.2f})",
        "internal_marks": f"internal marks ({data.internal_marks:.1f})",
        "backlogs":       f"backlogs ({data.backlogs})",
    }

    for feat, _ in ranked[:2]:
        top_factors.append(readable[feat])
        if feat == "attendance" and data.attendance < 65:
            reasons.append("low attendance")
        elif feat == "prev_gpa" and data.prev_gpa < 5:
            reasons.append("declining GPA")
        elif feat == "internal_marks" and data.internal_marks < 45:
            reasons.append("poor internal marks")
        elif feat == "backlogs" and data.backlogs >= 3:
            reasons.append("high number of backlogs")

    if reasons:
        explanation = f"{risk} risk primarily due to {' and '.join(reasons)}."
    else:
        explanation = f"{risk} risk based on overall academic performance profile."

    return explanation, top_factors

# ── Helper: CSV logger ────────────────────────────────────────────────────────
def log_prediction(data: StudentInput, risk: str, confidence: float, alerts: list):
    header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow([
                "timestamp", "student_name", "attendance", "prev_gpa",
                "internal_marks", "backlogs", "risk_level", "confidence", "alerts"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            data.student_name,
            data.attendance,
            data.prev_gpa,
            data.internal_marks,
            data.backlogs,
            risk,
            round(confidence, 4),
            " | ".join(alerts)
        ])

# ── Helper: Email alert ───────────────────────────────────────────────────────
def send_email_alert(to_email: str, student: str, risk: str,
                     alerts: list, explanation: str):
    """
    Sends an HTML email alert. 
    Configure SMTP_* env variables or replace hard-coded defaults.
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not smtp_user or not smtp_pass:
        logger.warning("SMTP credentials not configured — skipping email.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 HIGH RISK ALERT — {student}"
    msg["From"]    = smtp_user
    msg["To"]      = to_email

    alert_html = "".join(f"<li>{a}</li>" for a in alerts)
    html = f"""
    <html><body style="font-family:sans-serif;background:#1a1a2e;color:#eee;padding:20px">
      <h2 style="color:#ff4d6d">🚨 Student Risk Alert</h2>
      <p><strong>Student:</strong> {student}</p>
      <p><strong>Risk Level:</strong> <span style="color:#ff4d6d">{risk}</span></p>
      <p><strong>Explanation:</strong> {explanation}</p>
      <h3>Alerts</h3><ul>{alert_html}</ul>
      <h3>Suggestions</h3>
      <ul>
        <li>Schedule an immediate counselling session</li>
        <li>Provide supplementary study materials</li>
        <li>Monitor attendance closely for the next 2 weeks</li>
        <li>Connect with a faculty mentor</li>
      </ul>
    </body></html>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        logger.info(f"Alert email sent to {to_email}")
    except Exception as e:
        logger.error(f"Email failed: {e}")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Student Performance Prediction API is running ✅"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentInput, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run model/train_model.py first.")

    X = np.array([[data.attendance, data.prev_gpa,
                   data.internal_marks, data.backlogs]])
    X_scaled = scaler.transform(X)

    proba      = model.predict_proba(X_scaled)[0]
    pred_idx   = int(np.argmax(proba))
    risk_level = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    alerts            = generate_alerts(data)
    explanation, tops = generate_explanation(data, risk_level, importances)

    # Log asynchronously
    background_tasks.add_task(log_prediction, data, risk_level, confidence, alerts)

    # Email if HIGH risk and email provided
    if risk_level == "High" and data.email:
        background_tasks.add_task(
            send_email_alert, data.email, data.student_name,
            risk_level, alerts, explanation
        )

    logger.info(f"Prediction: {data.student_name} → {risk_level} ({confidence:.2%})")

    return PredictionResponse(
        student_name=data.student_name,
        risk_level=risk_level,
        confidence=round(confidence, 4),
        alerts=alerts,
        explanation=explanation,
        top_factors=tops
    )


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")

    content = await file.read()
    with open(UPLOAD_FILE, "wb") as f:
        f.write(content)

    logger.info(f"Uploaded file saved: {UPLOAD_FILE}")
    return {"message": "File uploaded successfully.", "filename": file.filename}


@app.post("/retrain")
def retrain():
    """Merge original + uploaded data, retrain, reload model."""
    original = os.path.join(DATA_DIR, "student_data.csv")
    merged   = os.path.join(DATA_DIR, "merged_data.csv")

    if not os.path.exists(original):
        raise HTTPException(404, "Original dataset not found.")

    df_orig = pd.read_csv(original)

    if os.path.exists(UPLOAD_FILE):
        df_new = pd.read_csv(UPLOAD_FILE)
        required = {"attendance", "prev_gpa", "internal_marks", "backlogs", "risk_level"}
        if not required.issubset(df_new.columns):
            raise HTTPException(422, f"Uploaded CSV must have columns: {required}")
        df_merged = pd.concat([df_orig, df_new], ignore_index=True)
    else:
        df_merged = df_orig

    df_merged.to_csv(merged, index=False)
    logger.info(f"Merged dataset: {len(df_merged)} rows → {merged}")

    # Run training script as subprocess
    train_script = os.path.join(MODEL_DIR, "train_model.py")
    # Patch DATA path via env var override (simple approach)
    env = os.environ.copy()
    env["STUDENT_DATA_PATH"] = merged

    result = subprocess.run(
        [sys.executable, train_script],
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        logger.error(result.stderr)
        raise HTTPException(500, f"Retraining failed:\n{result.stderr[:500]}")

    # Reload artifacts
    global model, scaler, le, importances
    model, scaler, le, importances = load_artifacts()
    logger.info("Model reloaded after retraining.")

    return {
        "message":  "Model retrained and reloaded successfully.",
        "rows_used": len(df_merged),
        "output":   result.stdout[-800:]
    }


@app.get("/logs")
def get_logs(limit: int = 100):
    if not os.path.exists(LOG_FILE):
        return {"logs": [], "total": 0}

    df = pd.read_csv(LOG_FILE)
    df = df.tail(limit).fillna("")
    return {"logs": df.to_dict(orient="records"), "total": len(df)}


@app.get("/stats")
def get_stats():
    """Quick stats from the log file for the analytics page."""
    if not os.path.exists(LOG_FILE):
        return {"total": 0}

    df = pd.read_csv(LOG_FILE)
    return {
        "total":        len(df),
        "risk_counts":  df["risk_level"].value_counts().to_dict(),
        "avg_confidence": round(df["confidence"].mean(), 4) if "confidence" in df else None,
    }

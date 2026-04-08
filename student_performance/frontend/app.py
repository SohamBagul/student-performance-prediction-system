"""
app.py — Streamlit Frontend Dashboard
======================================
Dark-themed dashboard with 4 pages:
  1. Prediction
  2. Analytics
  3. Upload & Retrain
  4. Logs
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ── Config ────────────────────────────────────────────────────────────────────
API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Student Risk Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0d0f1a;
    color: #e2e8f0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1123 0%, #141729 100%);
    border-right: 1px solid #1e2340;
  }

  /* Cards */
  .metric-card {
    background: linear-gradient(135deg, #13162b 0%, #1a1f38 100%);
    border: 1px solid #2a3060;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }

  /* Risk badges */
  .risk-high   { background:#ff4d6d22; color:#ff4d6d; border:1px solid #ff4d6d; }
  .risk-medium { background:#ffd16022; color:#ffd160; border:1px solid #ffd160; }
  .risk-low    { background:#4ade8022; color:#4ade80; border:1px solid #4ade80; }

  .risk-badge {
    display:inline-block; padding:6px 20px; border-radius:999px;
    font-weight:700; font-size:1.1rem; letter-spacing:1px;
    font-family:'JetBrains Mono', monospace;
  }

  /* Alert box */
  .alert-box {
    background:#ff4d6d11; border:1px solid #ff4d6d44;
    border-left:4px solid #ff4d6d; border-radius:8px;
    padding:12px 16px; margin:6px 0; font-size:0.9rem;
  }

  /* Explanation box */
  .explain-box {
    background:#6c63ff11; border:1px solid #6c63ff44;
    border-left:4px solid #6c63ff; border-radius:8px;
    padding:12px 16px; margin-top:12px; font-size:0.95rem;
  }

  /* Section headers */
  .section-title {
    font-size:1.6rem; font-weight:700; color:#a5b4fc;
    margin-bottom:8px; letter-spacing:-0.5px;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #6c63ff, #4f46e5);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 12px 28px;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px #6c63ff55;
  }

  /* Inputs */
  .stNumberInput > div > div > input,
  .stTextInput > div > div > input {
    background: #1a1f38; border: 1px solid #2a3060;
    border-radius: 8px; color: #e2e8f0;
  }

  /* Tabs */
  .stTabs [role="tab"] { color:#94a3b8; font-weight:500; }
  .stTabs [aria-selected="true"] { color:#a5b4fc; border-bottom-color:#6c63ff !important; }

  /* Divider */
  hr { border-color: #1e2340; }

  /* Scrollbar */
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:#0d0f1a; }
  ::-webkit-scrollbar-thumb { background:#2a3060; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 30px">
      <div style="font-size:2.5rem">🎓</div>
      <div style="font-size:1.2rem;font-weight:700;color:#a5b4fc">Student Risk Monitor</div>
      <div style="font-size:0.75rem;color:#64748b;margin-top:4px">Early Warning System</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🔮 Prediction", "📊 Analytics", "📤 Upload & Retrain", "📋 Logs"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#475569;padding:8px">
      <div>Backend: <code style="color:#6c63ff">localhost:8000</code></div>
      <div style="margin-top:4px">Model: <code style="color:#6c63ff">XGBoost / RF</code></div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Prediction":
    st.markdown('<div class="section-title">🔮 Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter student details below to predict academic risk level.")
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Student Information")
        name  = st.text_input("Student Name", value="Enter Your Name")
        email = st.text_input("Email (optional — for HIGH risk alerts)", placeholder="En.No@students.mituniversity.edu.in")

        st.markdown("#### Academic Data")
        c1, c2 = st.columns(2)
        with c1:
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 72.0, 0.5)
            internal   = st.number_input("Internal Marks (0–100)", 0.0, 100.0, 55.0, 0.5)
        with c2:
            gpa      = st.number_input("Previous GPA (0–10)", 0.0, 10.0, 6.5, 0.1)
            backlogs = st.number_input("Backlogs", 0, 20, 1, 1)

        predict_btn = st.button("🔍 Predict Risk Level", use_container_width=True)

    with col2:
        if predict_btn:
            payload = {
                "student_name":   name,
                "email":          email or None,
                "attendance":     attendance,
                "prev_gpa":       gpa,
                "internal_marks": internal,
                "backlogs":       int(backlogs)
            }
            with st.spinner("Analysing student profile …"):
                try:
                    r = requests.post(f"{API}/predict", json=payload, timeout=10)
                    if r.status_code == 200:
                        res = r.json()
                        risk = res["risk_level"]
                        risk_class = f"risk-{risk.lower()}"

                        st.markdown("#### Prediction Result")
                        st.markdown(f"""
                        <div class="metric-card">
                          <div style="font-size:0.8rem;color:#64748b;text-transform:uppercase;letter-spacing:2px">Risk Level</div>
                          <div style="margin:10px 0">
                            <span class="risk-badge {risk_class}">{risk.upper()} RISK</span>
                          </div>
                          <div style="font-size:0.85rem;color:#94a3b8;margin-top:8px">
                            Confidence: <strong style="color:#e2e8f0">{res['confidence']*100:.1f}%</strong>
                          </div>
                          <div style="margin-top:12px;font-size:0.8rem;color:#64748b">Student: {name}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if res["alerts"]:
                            st.markdown("#### ⚠️ Alerts")
                            for a in res["alerts"]:
                                st.markdown(f'<div class="alert-box">{a}</div>', unsafe_allow_html=True)

                        st.markdown(f'<div class="explain-box">💡 {res["explanation"]}</div>',
                                    unsafe_allow_html=True)

                        if res.get("top_factors"):
                            st.markdown("#### Top Contributing Factors")
                            for i, f in enumerate(res["top_factors"], 1):
                                st.markdown(f"**{i}.** {f}")
                    else:
                        st.error(f"API error {r.status_code}: {r.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the backend. Is the FastAPI server running?")
        else:
            st.markdown("""
            <div class="metric-card" style="text-align:center;padding:40px">
              <div style="font-size:3rem;margin-bottom:12px">🎯</div>
              <div style="color:#64748b">Fill in the form and click<br><strong style="color:#a5b4fc">Predict Risk Level</strong></div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    try:
        r = requests.get(f"{API}/logs", params={"limit": 500}, timeout=5)
        logs = r.json().get("logs", [])
    except:
        logs = []

    if not logs:
        st.info("No prediction logs yet. Make some predictions first!")
    else:
        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["confidence"] = df["confidence"].astype(float)

        # ── KPI row ────────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        total = len(df)
        high  = (df["risk_level"] == "High").sum()
        med   = (df["risk_level"] == "Medium").sum()
        low   = (df["risk_level"] == "Low").sum()

        for col, label, value, color in [
            (k1, "Total Predictions", total,   "#a5b4fc"),
            (k2, "🔴 High Risk",      high,    "#ff4d6d"),
            (k3, "🟡 Medium Risk",    med,     "#ffd160"),
            (k4, "🟢 Low Risk",       low,     "#4ade80"),
        ]:
            col.markdown(f"""
            <div class="metric-card" style="text-align:center">
              <div style="font-size:2rem;font-weight:700;color:{color}">{value}</div>
              <div style="font-size:0.8rem;color:#64748b;margin-top:4px">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        row1a, row1b = st.columns(2)

        # Risk Distribution Donut
        with row1a:
            risk_counts = df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk", "Count"]
            colors = {"High":"#ff4d6d", "Medium":"#ffd160", "Low":"#4ade80"}
            fig = px.pie(
                risk_counts, names="Risk", values="Count",
                title="Risk Level Distribution",
                color="Risk", color_discrete_map=colors,
                hole=0.55
            )
            fig.update_layout(
                paper_bgcolor="#13162b", plot_bgcolor="#13162b",
                font_color="#e2e8f0", title_font_size=14,
                legend=dict(bgcolor="#13162b")
            )
            st.plotly_chart(fig, use_container_width=True)

        # Attendance vs Risk
        with row1b:
            fig2 = px.box(
                df, x="risk_level", y="attendance",
                color="risk_level",
                color_discrete_map=colors,
                title="Attendance vs Risk Level",
                labels={"attendance": "Attendance (%)", "risk_level": "Risk Level"}
            )
            fig2.update_layout(
                paper_bgcolor="#13162b", plot_bgcolor="#1a1f38",
                font_color="#e2e8f0", showlegend=False,
                title_font_size=14
            )
            fig2.update_xaxes(gridcolor="#1e2340")
            fig2.update_yaxes(gridcolor="#1e2340")
            st.plotly_chart(fig2, use_container_width=True)

        row2a, row2b = st.columns(2)

        # GPA Trend
        with row2a:
            df_sorted = df.sort_values("timestamp")
            fig3 = px.scatter(
                df_sorted, x="prev_gpa", y="attendance",
                color="risk_level", color_discrete_map=colors,
                title="GPA vs Attendance (coloured by Risk)",
                labels={"prev_gpa": "GPA", "attendance": "Attendance (%)"}
            )
            fig3.update_layout(
                paper_bgcolor="#13162b", plot_bgcolor="#1a1f38",
                font_color="#e2e8f0", title_font_size=14
            )
            fig3.update_xaxes(gridcolor="#1e2340")
            fig3.update_yaxes(gridcolor="#1e2340")
            st.plotly_chart(fig3, use_container_width=True)

        # Confidence over time
        with row2b:
            fig4 = px.line(
                df.sort_values("timestamp"),
                x="timestamp", y="confidence",
                title="Model Confidence Over Time",
                labels={"confidence": "Confidence", "timestamp": "Time"}
            )
            fig4.update_traces(line_color="#6c63ff", line_width=2)
            fig4.update_layout(
                paper_bgcolor="#13162b", plot_bgcolor="#1a1f38",
                font_color="#e2e8f0", title_font_size=14
            )
            fig4.update_xaxes(gridcolor="#1e2340")
            fig4.update_yaxes(gridcolor="#1e2340")
            st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — UPLOAD & RETRAIN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload & Retrain":
    st.markdown('<div class="section-title">📤 Upload & Retrain</div>', unsafe_allow_html=True)
    st.markdown("Upload a labelled CSV file to augment the training data and retrain the model.")
    st.markdown("---")

    st.markdown("""
    <div class="metric-card">
      <strong>Expected CSV columns:</strong>
      <code style="color:#a5b4fc;display:block;margin-top:8px;font-size:0.85rem">
        attendance, prev_gpa, internal_marks, backlogs, risk_level
      </code>
      <div style="font-size:0.8rem;color:#64748b;margin-top:8px">
        risk_level must be one of: <em>Low, Medium, High</em>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        df_preview = pd.read_csv(uploaded)
        st.markdown(f"**Preview** ({len(df_preview)} rows)")
        st.dataframe(df_preview.head(10), use_container_width=True)
        uploaded.seek(0)  # reset pointer

        col_up, col_rt = st.columns(2)
        with col_up:
            if st.button("📤 Upload File", use_container_width=True):
                with st.spinner("Uploading …"):
                    try:
                        r = requests.post(
                            f"{API}/upload-data",
                            files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                            timeout=15
                        )
                        if r.status_code == 200:
                            st.success("✅ File uploaded successfully!")
                        else:
                            st.error(f"Upload failed: {r.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend.")

        with col_rt:
            if st.button("🔄 Retrain Model", use_container_width=True):
                with st.spinner("Retraining model — this may take a minute …"):
                    try:
                        r = requests.post(f"{API}/retrain", timeout=120)
                        if r.status_code == 200:
                            res = r.json()
                            st.success(f"✅ {res['message']}")
                            st.code(res.get("output", ""), language="text")
                        else:
                            st.error(f"Retrain failed: {r.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LOGS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Logs":
    st.markdown('<div class="section-title">📋 Prediction Logs</div>', unsafe_allow_html=True)
    st.markdown("---")

    try:
        r    = requests.get(f"{API}/logs", params={"limit": 200}, timeout=5)
        data = r.json()
        logs = data.get("logs", [])
        total = data.get("total", 0)
    except:
        logs = []
        total = 0

    if not logs:
        st.info("No logs available yet.")
    else:
        st.markdown(f"Showing last **{len(logs)}** of **{total}** predictions.")
        df = pd.DataFrame(logs)

        # Colour-code risk level
        def highlight_risk(row):
            colors = {"High": "background-color:#ff4d6d22",
                      "Medium": "background-color:#ffd16022",
                      "Low": "background-color:#4ade8022"}
            return [colors.get(row.get("risk_level",""), "")] * len(row)

        styled = df.style.apply(highlight_risk, axis=1)
        st.dataframe(styled, use_container_width=True, height=500)

        # Download
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Logs CSV",
            data=csv_bytes,
            file_name=f"prediction_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

import streamlit as ts
import panadas as pd
import plotly.express as px 
import numpy as np
import hashlib
import requests
from sklearn.ensemble import IsolationForest

#===============
#app config
#===============

st.set_page_config(
    page_title="Clinical Digital Health Dashboard",
    layout="wide"
)

st.title("🩺 Clinical Digital Health Monitoring Dashboard")
st.markdown("""
**Early risk detection for shock/sepsis, hypertensive emergencies, and hyperglycemia**  
*IoT sensors monitors
""")

# =========================
# LOAD & PREPROCESS DATA
# =========================
@st.cache_data
def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Normalize columns
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    # Patients Privacy - Anonymization
    df["patient_id_hash"] = df["patient_id"].astype(str).apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()
    )
    df = df.drop(columns=["patient_id"])

    df = df.set_index("timestamp")

    # Selective interpolation
    interp_cols = ["heart_rate", "respiratory_rate", "body_temperature"]
    df[interp_cols] = df[interp_cols].interpolate(method="time", limit=2)

    # Measurement density
    df["hr_count_30min"] = df["heart_rate"].notna().rolling("30min").sum()
    df["hr_reliable"] = df["hr_count_30min"] >= 3

    # Rolling HR
    df["hr_rolling_30min"] = df["heart_rate"].rolling("30min").mean()
    df.loc[~df["hr_reliable"], "hr_rolling_30min"] = np.nan

    return df

df = load_and_preprocess("data/iot_health_monitoring_dataset.csv")

# =========================
# HR ANOMALY FEATURE
# =========================
df_ml = df.reset_index().copy()
df_ml["row_id"] = df_ml.index

features = df_ml[["row_id", "heart_rate", "activity_level", "hrv_sdnn"]].dropna()

iso = IsolationForest(contamination=0.05, random_state=42)
features["hr_anomaly"] = iso.fit_predict(
    features[["heart_rate", "activity_level", "hrv_sdnn"]]
)

df_ml = df_ml.merge(
    features[["row_id", "hr_anomaly"]],
    on="row_id",
    how="left"
)

df = df_ml.drop(columns="row_id").set_index("timestamp")

# =========================================================================
# CLINICAL RISK LOGIC:SEPSIS, HYPERTENSIVE EMERGENCY, HYPERG/HYPOGLYCEMIA
# =========================================================================
df["shock_sepsis_risk"] = (
    (df["heart_rate"] > 90) & (df["respiratory_rate"] > 20) & (df['body_temperature'] > 100.4) |
    (df["hr_anomaly"] == -1)
)

df["htn_emergency_risk"] = (
    (df["blood_pressure_systolic"] >= 180) |
    (df["blood_pressure_diastolic"] >= 120)
)

df["hyperglycemia"] = df["glucose_level"] >= 180
df["severe_hyperglycemia"] = df["glucose_level"] >= 250

# =========================
# RISK BADGE FUNCTION
# =========================
def risk_badge(condition):
    if condition:
        return "🔴 HIGH"
    return "🟢 LOW"

# =========================
# TOP KPIs
# =========================
latest = df.iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Shock / Sepsis Risk",
    risk_badge(latest["shock_sepsis_risk"])
)

col2.metric(
    "Hypertensive Emergency",
    risk_badge(latest["htn_emergency_risk"])
)

col3.metric(
    "Hyperglycemia",
    "🔴 SEVERE" if latest["severe_hyperglycemia"]
    else ("🟠 PRESENT" if latest["hyperglycemia"] else "🟢 NORMAL")
)

# =========================
# PANEL 1 — SHOCK / SEPSIS
# =========================
st.subheader("🔴 Early Shock / Sepsis Risk")

fig = px.line(
    df.reset_index(),
    x="timestamp",
    y=["heart_rate", "respiratory_rate"],
    title="Heart Rate & Respiratory Rate"
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# PANEL 2 — BLOOD PRESSURE
# =========================
st.subheader("🔵 Hypertensive Emergency Risk")

fig = px.line(
    df.reset_index(),
    x="timestamp",
    y=["blood_pressure_systolic", "blood_pressure_diastolic"],
    title="Blood Pressure Trends"
)

fig.add_hline(y=180, line_dash="dash", line_color="red")
fig.add_hline(y=120, line_dash="dash", line_color="red")

st.plotly_chart(fig, use_container_width=True)

# =========================
# PANEL 3 — GLUCOSE
# =========================
st.subheader("🟠 Hyperglycemia & Metabolic Risk")

fig = px.line(
    df.reset_index(),
    x="timestamp",
    y="glucose_level",
    title="Glucose Levels Over Time"
)

fig.add_hline(y=180, line_dash="dash", line_color="orange")
fig.add_hline(y=250, line_dash="dash", line_color="red")

st.plotly_chart(fig, use_container_width=True)

# =========================
# PANEL 4 — CONTEXT
# =========================
st.subheader("🟢 Behavioral & Modifiable Factors")

fig = px.scatter(
    df,
    x="activity_level",
    y="heart_rate",
    color="shock_sepsis_risk",
    title="Heart Rate vs Activity Level"
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# PANEL 5 — PATIENT EDUCATION
# =========================
st.subheader("📘 Patient Education (Actionable & Modifiable)")

education = []

if latest["sleep_quality"] < 0.3:
    education.append("Improve sleep consistency — poor sleep increases cardiovascular stress.")

if latest["stress_level"] > 0.7:
    education.append("Stress reduction (breathing, mindfulness) may help lower heart rate.")

if latest["steps_count"] < 100:
    education.append("Gradually increase daily physical activity to improve BP and glucose control.")

if latest["hyperglycemia"]:
    education.append("Review diet and medication adherence to improve glucose control.")

if not education:
    education.append("Current lifestyle indicators are supportive of stable health.")

for msg in education:
    st.markdown(f"- {msg}")

# =========================
# FOOTER
# =========================
st.markdown("""
---
*This dashboard help to detect early signs of clinical deterioration*
""")

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import timedelta
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest
import plotly.express as px

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Clinical Alert Dashboard",
    layout="wide"
)

st.title("🩺 Mini - Clinical Digital Health Alert Dashboard")
st.caption("Severity-based, trend-aware medical risk monitoring")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    # Hash patient ID, keep device_id
    df["patient_id_hash"] = df["patient_id"].astype(str).apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()
    )
    df = df.drop(columns=["patient_id"])

    df = df.set_index("timestamp")
    return df


df = load_data("data/iot_health_monitoring_dataset.csv")

# =====================================================
# PREPROCESSING & RELIABILITY
# =====================================================
INTERPOLATE_COLS = ["heart_rate", "respiratory_rate", "body_temperature"]

df[INTERPOLATE_COLS] = df[INTERPOLATE_COLS].interpolate(
    method="time", limit=2
)

df["hr_count_30min"] = df["heart_rate"].notna().rolling("30min").sum()
df["hr_reliable"] = df["hr_count_30min"] >= 3

# =====================================================
# ML: HR ANOMALY (SUPPORTING SIGNAL ONLY)
# =====================================================
df_ml = df.reset_index().copy()
df_ml["row_id"] = df_ml.index

ml_features = df_ml[
    ["row_id", "heart_rate", "activity_level", "hrv_sdnn"]
].dropna()

iso = IsolationForest(contamination=0.05, random_state=42)
ml_features["hr_anomaly"] = iso.fit_predict(
    ml_features[["heart_rate", "activity_level", "hrv_sdnn"]]
)

df_ml = df_ml.merge(
    ml_features[["row_id", "hr_anomaly"]],
    on="row_id",
    how="left"
)

df = df_ml.drop(columns="row_id").set_index("timestamp")

# =====================================================
# CLINICAL SCORING FUNCTIONS
# =====================================================
def sepsis_score(row):
    score = 0
    reasons = []

    if row["heart_rate"] > 100:
        score += 1; reasons.append("Tachycardia")
    if row["respiratory_rate"] > 22:
        score += 1; reasons.append("Tachypnea")
    if row["body_temperature"] > 100.4:
        score += 1; reasons.append("Fever")
    if row.get("hr_anomaly") == -1:
        score += 1; reasons.append("HR anomaly")

    return score, reasons


def stratify(score):
    if score >= 3:
        return "HIGH"
    if score == 2:
        return "MEDIUM"
    if score == 1:
        return "LOW"
    return "NONE"


def hypertension_severity(row):
    sbp = row["blood_pressure_systolic"]
    dbp = row["blood_pressure_diastolic"]

    if sbp >= 180 or dbp >= 120:
        return "EMERGENCY", ["Severely elevated BP"]
    elif sbp >= 140 or dbp >= 90:
        return "STAGE_2", ["Stage 2 hypertension"]
    elif sbp >= 130 or dbp >= 80:
        return "STAGE_1", ["Stage 1 hypertension"]
    else:
        return "NORMAL", []


def glycemic_risk(row):
    g = row["glucose_level"]

    if g < 70:
        return "HYPOGLYCEMIA", ["Low glucose"]
    elif g >= 250:
        return "SEVERE_HYPERGLYCEMIA", ["Severely elevated glucose"]
    elif g >= 180:
        return "HYPERGLYCEMIA", ["Elevated glucose"]
    else:
        return "NORMAL", []


# =====================================================
# APPLY CLINICAL LOGIC
# =====================================================
df[["sepsis_score", "sepsis_reasons"]] = df.apply(
    lambda r: pd.Series(sepsis_score(r)), axis=1
)
df["sepsis_severity"] = df["sepsis_score"].apply(stratify)

df[["htn_severity", "htn_reasons"]] = df.apply(
    lambda r: pd.Series(hypertension_severity(r)), axis=1
)

df[["glycemic_severity", "glycemic_reasons"]] = df.apply(
    lambda r: pd.Series(glycemic_risk(r)), axis=1
)

# =====================================================
# TREND-BASED WORSENING (PATIENT-LEVEL)
# =====================================================
def rolling_slope(series, window="30min"):
    slopes = []
    for i in range(len(series)):
        window_data = series.iloc[:i+1].last(window)
        if len(window_data) < 3:
            slopes.append(np.nan)
            continue
        x = np.arange(len(window_data))
        y = window_data.values
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)
    return slopes


df["hr_trend"] = (
    df.groupby("patient_id_hash", group_keys=False)["heart_rate"]
    .apply(rolling_slope)
)

df["sbp_trend"] = (
    df.groupby("patient_id_hash", group_keys=False)["blood_pressure_systolic"]
    .apply(rolling_slope)
)

df["glucose_trend"] = (
    df.groupby("patient_id_hash", group_keys=False)["glucose_level"]
    .apply(rolling_slope)
)

def worsening_flags(row):
    flags = []
    if pd.notna(row["hr_trend"]) and row["hr_trend"] > 0.5:
        flags.append("Heart rate rising")
    if pd.notna(row["sbp_trend"]) and row["sbp_trend"] > 0.5:
        flags.append("Blood pressure rising")
    if pd.notna(row["glucose_trend"]) and row["glucose_trend"] > 1:
        flags.append("Glucose rising")
    return flags


df["worsening_flags"] = df.apply(worsening_flags, axis=1)
df["is_worsening"] = df["worsening_flags"].apply(lambda x: len(x) > 0)

# Escalation
df["sepsis_severity_final"] = df["sepsis_severity"]
df.loc[
    (df["sepsis_severity"] == "MEDIUM") & (df["is_worsening"]),
    "sepsis_severity_final"
] = "HIGH"

df["htn_severity_final"] = df["htn_severity"]
df.loc[
    (df["htn_severity"] == "STAGE_2") & (df["is_worsening"]),
    "htn_severity_final"
] = "EMERGENCY"

# =====================================================
# ALERT EVENT GENERATION (WITH COOLDOWN)
# =====================================================
ALERT_COOLDOWN = timedelta(minutes=30)
alert_events = []
last_alert_time = {}

for ts, row in df.iterrows():
    patient = row["patient_id_hash"]
    device = row["device_id"]

    # SEPSIS
    if row["sepsis_severity_final"] in ["HIGH", "MEDIUM"] and row["hr_reliable"]:
        key = (patient, "SEPSIS")
        if key not in last_alert_time or ts - last_alert_time[key] > ALERT_COOLDOWN:
            reasons = row["sepsis_reasons"] + row["worsening_flags"]
            alert_events.append({
                "timestamp": ts,
                "patient_id_hash": patient,
                "device_id": device,
                "alert_type": "Shock / Sepsis",
                "severity": row["sepsis_severity_final"],
                "reason": ", ".join(reasons)
            })
            last_alert_time[key] = ts

    # HYPERTENSION
    if row["htn_severity_final"] in ["STAGE_2", "EMERGENCY"]:
        key = (patient, "HTN")
        if key not in last_alert_time or ts - last_alert_time[key] > ALERT_COOLDOWN:
            reasons = row["htn_reasons"] + row["worsening_flags"]
            alert_events.append({
                "timestamp": ts,
                "patient_id_hash": patient,
                "device_id": device,
                "alert_type": "Hypertension",
                "severity": row["htn_severity_final"],
                "reason": ", ".join(reasons)
            })
            last_alert_time[key] = ts

    # GLYCEMIC
    if row["glycemic_severity"] in ["HYPOGLYCEMIA", "SEVERE_HYPERGLYCEMIA"]:
        key = (patient, "GLUCOSE")
        if key not in last_alert_time or ts - last_alert_time[key] > ALERT_COOLDOWN:
            alert_events.append({
                "timestamp": ts,
                "patient_id_hash": patient,
                "device_id": device,
                "alert_type": "Glycemic Emergency",
                "severity": row["glycemic_severity"],
                "reason": ", ".join(row["glycemic_reasons"])
            })
            last_alert_time[key] = ts

alerts_df = pd.DataFrame(alert_events)

# =====================================================
# UI — ACTIVE ALERTS
# =====================================================
st.subheader("🚨 Active Clinical Alerts")

if alerts_df.empty:
    st.success("No active alerts")
else:
    st.dataframe(
        alerts_df.sort_values("timestamp", ascending=False),
        use_container_width=True
    )

# =====================================================
# UI — PATIENT VIEW
# =====================================================
st.subheader("👤 Patient Drill-down")

patient_ids = df["patient_id_hash"].unique()
selected_patient = st.selectbox("Select patient", patient_ids)

patient_df = df[df["patient_id_hash"] == selected_patient]

fig = px.line(
    patient_df.reset_index(),
    x="timestamp",
    y="heart_rate",
    color="sepsis_severity_final",
    title="Heart Rate with Sepsis Severity",
    color_discrete_map={
        "NONE": "green",
        "LOW": "yellow",
        "MEDIUM": "orange",
        "HIGH": "red"
    }
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# UI — RISK HEATMAP
# =====================================================
risk_df = patient_df.copy()

risk_df["Shock / Sepsis"] = risk_df["sepsis_severity_final"].map(
    {"NONE": 0, "LOW": 0, "MEDIUM": 1, "HIGH": 2}
)

risk_df["HTN Emergency"] = risk_df["htn_severity_final"].map(
    {"NORMAL": 0, "STAGE_1": 0, "STAGE_2": 1, "EMERGENCY": 2}
)

risk_df["Hyperglycemia"] = risk_df["glycemic_severity"].map(
    {"NORMAL": 0, "HYPERGLYCEMIA": 1, "SEVERE_HYPERGLYCEMIA": 2, "HYPOGLYCEMIA": 2}
)

risk_df["HR Anomaly"] = (risk_df["hr_anomaly"] == -1).astype(int)

heatmap_df = risk_df[
    ["Shock / Sepsis", "HTN Emergency", "Hyperglycemia", "HR Anomaly"]
].T

fig = px.imshow(
    heatmap_df,
    aspect="auto",
    color_continuous_scale=[[0, "green"], [0.5, "orange"], [1, "red"]],
    title="Clinical Risk Timeline"
)

st.plotly_chart(fig, use_container_width=True)

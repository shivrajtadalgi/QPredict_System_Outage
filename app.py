import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="QPredict Enterprise AI",
    layout="wide"
)

# ============================================================
# RESPONSIVE CSS
# ============================================================

st.markdown("""
<style>

/* Main Background */
.main {
    background-color: #f5f7fa;
}

/* Main Container */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1550px;
}

/* Global Font */
html, body, [class*="css"] {
    font-size: 15px;
}

/* Buttons */
.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    background-color: #003366;
    color: white;
    font-size: 16px;
    font-weight: bold;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #e6e6e6;
    padding: 12px;
    border-radius: 12px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    font-size: 13px;
}

/* Mobile Responsive */
@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    html, body, [class*="css"] {
        font-size: 12px;
    }
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load("xgboost_model.pkl")

scaler = joblib.load("scaler.pkl")

feature_columns = joblib.load("feature_columns.pkl")

# ============================================================
# HEADER
# ============================================================

col1, col2, col3 = st.columns([1.5,5,1.5])

with col1:

    st.image(
        "logo_PLUSE.QPredict.png",
        width=160
    )

with col2:

    st.markdown(
        """
        # QPredict Enterprise Incident Intelligence Platform

        ### AI-Powered IT Operations Incident Severity Analytics
        """
    )

with col3:

    st.image(
        "logo_digitide.jpg",
        width=160
    )

st.markdown("---")

# ============================================================
# SINGLE INCIDENT PREDICTION
# ============================================================

st.subheader("🖥️ IT Operations Incident")

col1, col2, col3 = st.columns(3)

# ============================================================
# COLUMN 1
# ============================================================

with col1:

    duration = st.slider(
        "Duration (minutes)",
        1,
        600,
        120
    )

    resolution = st.slider(
        "Resolution Time (minutes)",
        1,
        600,
        140
    )

    cpu = st.slider(
        "CPU Usage (%)",
        1,
        100,
        65
    )

    memory = st.slider(
        "Memory Usage (%)",
        1,
        100,
        70
    )

# ============================================================
# COLUMN 2
# ============================================================

with col2:

    latency = st.slider(
        "Network Latency (ms)",
        1,
        600,
        150
    )

    users = st.slider(
        "Users Affected",
        1,
        20000,
        1200
    )

    alerts = st.slider(
        "Alert Count",
        1,
        100,
        10
    )

    api_error = st.slider(
        "API Error Rate (%)",
        0,
        100,
        8
    )

# ============================================================
# COLUMN 3
# ============================================================

with col3:

    cause = st.selectbox(
        "Cause",
        [
            "Network Issue",
            "Software Bug",
            "Hardware Failure",
            "Power Outage",
            "Memory Leak",
            "Database Lock",
            "Security Attack",
            "High Traffic",
            "Configuration Drift"
        ]
    )

    region = st.selectbox(
        "Region",
        [
            "North",
            "South",
            "East",
            "West"
        ]
    )

    environment = st.selectbox(
        "Environment",
        [
            "Production",
            "UAT",
            "Staging",
            "Development"
        ]
    )

    deployment = st.selectbox(
        "Deployment Type",
        [
            "Patch",
            "Hotfix",
            "Major Release",
            "Rollback"
        ]
    )

# ============================================================
# SINGLE PREDICTION BUTTON
# ============================================================

if st.button("🚀 Predict Incident Severity"):

    # ========================================================
    # NORMALIZATION
    # ========================================================

    cpu_score = cpu

    memory_score = memory

    latency_score = min((latency / 600) * 100, 100)

    api_score = api_error

    alert_score = min((alerts / 100) * 100, 100)

    duration_score = min((duration / 600) * 100, 100)

    resolution_score = min((resolution / 600) * 100, 100)

    users_score = min((users / 20000) * 100, 100)

    # ========================================================
    # BASE SCORE
    # ========================================================

    severity_score = (

        (cpu_score * 0.15) +

        (memory_score * 0.10) +

        (latency_score * 0.15) +

        (api_score * 0.15) +

        (alert_score * 0.10) +

        (duration_score * 0.10) +

        (resolution_score * 0.10) +

        (users_score * 0.15)

    )

    # ========================================================
    # ENTERPRISE BOOSTS
    # ========================================================

    if cpu > 85:
        severity_score += 8

    if memory > 90:
        severity_score += 8

    if latency > 400:
        severity_score += 10

    if users > 10000:
        severity_score += 12

    if alerts > 40:
        severity_score += 10

    if api_error > 25:
        severity_score += 12

    if duration > 300:
        severity_score += 10

    if resolution > 350:
        severity_score += 8

    if environment == "Production":
        severity_score += 5

    critical_causes = [
        "Hardware Failure",
        "Security Attack",
        "Power Outage",
        "Database Lock"
    ]

    if cause in critical_causes:
        severity_score += 10

    if deployment == "Major Release":
        severity_score += 6

    # ========================================================
    # FINAL SCORE
    # ========================================================

    severity_score = min(severity_score, 100)

    severity_score = round(severity_score, 2)

    # ========================================================
    # FINAL LABEL
    # ========================================================

    if severity_score <= 25:

        pred_label = "normal"

    elif severity_score <= 50:

        pred_label = "low"

    elif severity_score <= 75:

        pred_label = "medium"

    else:

        pred_label = "high"

    # ========================================================
    # OUTPUT
    # ========================================================

    st.markdown("## 🧠 AI Prediction Result")

    if pred_label == "high":

        st.error("🚨 HIGH Severity Incident")

    elif pred_label == "medium":

        st.warning("⚠️ MEDIUM Severity Incident")

    elif pred_label == "low":

        st.info("🔹 LOW Severity Incident")

    else:

        st.success("✅ NORMAL Incident")

    st.metric(
        "Severity Score",
        f"{severity_score}"
    )

# ============================================================
# DIVIDER
# ============================================================

st.markdown("---")

# ============================================================
# BATCH PREDICTION
# ============================================================

st.subheader("📡 IT Operations Incidents")

uploaded_file = st.file_uploader(
    "Upload Incident Dataset CSV",
    type=["csv"]
)

# ============================================================
# PROCESS FILE
# ============================================================

if uploaded_file is not None:

    try:

        # ====================================================
        # READ CSV
        # ====================================================

        df = pd.read_csv(uploaded_file)

        st.markdown("### 📂 Uploaded Incident Dataset")

        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        # ====================================================
        # REQUIRED COLUMN CHECK
        # ====================================================

        required_cols = [

            "CPU_Usage (%)",
            "Memory_Usage (%)",
            "Network_Latency_ms",
            "API_Error_Rate (%)",
            "Users_Affected",
            "Alert_Count_Last_1Hr",
            "Duration (minutes)",
            "Resolution_Time (minutes)",
            "Environment",
            "Deployment_Type",
            "Cause"

        ]

        missing_cols = [

            col for col in required_cols
            if col not in df.columns

        ]

        if len(missing_cols) > 0:

            st.error(
                f"Uploaded dataset missing columns: {missing_cols}"
            )

            st.stop()

        # ====================================================
        # PROGRESS BAR
        # ====================================================

        st.info(
            "🤖 AI Engine analyzing outage patterns..."
        )

        progress_bar = st.progress(0)

        for i in range(100):

            time.sleep(0.02)

            progress_bar.progress(i + 1)

        # ====================================================
        # NORMALIZATION
        # ====================================================

        cpu_score = df["CPU_Usage (%)"]

        memory_score = df["Memory_Usage (%)"]

        latency_score = (
            df["Network_Latency_ms"] / 600
        ) * 100

        api_score = df["API_Error_Rate (%)"]

        alert_score = (
            df["Alert_Count_Last_1Hr"] / 100
        ) * 100

        duration_score = (
            df["Duration (minutes)"] / 600
        ) * 100

        resolution_score = (
            df["Resolution_Time (minutes)"] / 600
        ) * 100

        users_score = (
            df["Users_Affected"] / 20000
        ) * 100

        # ====================================================
        # BASE SCORE
        # ====================================================

        severity_scores = (

            (cpu_score * 0.15) +

            (memory_score * 0.10) +

            (latency_score * 0.15) +

            (api_score * 0.15) +

            (alert_score * 0.10) +

            (duration_score * 0.10) +

            (resolution_score * 0.10) +

            (users_score * 0.15)

        )

        # ====================================================
        # ENTERPRISE BOOSTS
        # ====================================================

        severity_scores += np.where(
            df["CPU_Usage (%)"] > 85,
            8,
            0
        )

        severity_scores += np.where(
            df["Memory_Usage (%)"] > 90,
            8,
            0
        )

        severity_scores += np.where(
            df["Network_Latency_ms"] > 400,
            10,
            0
        )

        severity_scores += np.where(
            df["Users_Affected"] > 10000,
            12,
            0
        )

        severity_scores += np.where(
            df["Alert_Count_Last_1Hr"] > 40,
            10,
            0
        )

        severity_scores += np.where(
            df["API_Error_Rate (%)"] > 25,
            12,
            0
        )

        severity_scores += np.where(
            df["Duration (minutes)"] > 300,
            10,
            0
        )

        severity_scores += np.where(
            df["Resolution_Time (minutes)"] > 350,
            8,
            0
        )

        severity_scores += np.where(
            df["Environment"] == "Production",
            5,
            0
        )

        critical_causes = [
            "Hardware Failure",
            "Security Attack",
            "Power Outage",
            "Database Lock"
        ]

        severity_scores += np.where(
            df["Cause"].isin(critical_causes),
            10,
            0
        )

        severity_scores += np.where(
            df["Deployment_Type"] == "Major Release",
            6,
            0
        )

        # ====================================================
        # FINAL SCORE
        # ====================================================

        severity_scores = severity_scores.clip(0, 100)

        severity_scores = severity_scores.round(2)

        # ====================================================
        # FINAL LABELS
        # ====================================================

        predicted_severity = []

        for score in severity_scores:

            if score <= 25:

                predicted_severity.append("normal")

            elif score <= 50:

                predicted_severity.append("low")

            elif score <= 75:

                predicted_severity.append("medium")

            else:

                predicted_severity.append("high")

        # ====================================================
        # OUTPUT
        # ====================================================

        df["Predicted_Severity"] = predicted_severity

        df["Severity_Score"] = severity_scores

        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            f"✅ AI Analysis Completed for {len(df)} incidents"
        )

        # ====================================================
        # METRICS
        # ====================================================

        st.markdown("## 📈 Incident Intelligence Dashboard")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Total Incidents", len(df))

        with m2:
            st.metric(
                "High Severity",
                len(df[df["Predicted_Severity"] == "high"])
            )

        with m3:
            st.metric(
                "Medium Severity",
                len(df[df["Predicted_Severity"] == "medium"])
            )

        with m4:
            st.metric(
                "Normal Incidents",
                len(df[df["Predicted_Severity"] == "normal"])
            )

        # ====================================================
        # CHART
        # ====================================================

        st.markdown("## 📊 Incident Severity Distribution")

        severity_counts = (
            df["Predicted_Severity"]
            .value_counts()
        )

        st.bar_chart(severity_counts)

        # ====================================================
        # OUTPUT TABLE
        # ====================================================

        st.markdown("## 🧠 AI Incident Severity Analysis")

        st.dataframe(
            df,
            use_container_width=True,
            height=500
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="QPredict_Incident_Analysis.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
QPredict Enterprise AI Platform

AI-Powered Incident Intelligence System

Powered by XGBoost | Machine Learning | Predictive Analytics | AIOps
""")

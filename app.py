import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="QPredict System Outage", layout="wide")

# ================================
# LOAD FILES
# ================================
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ================================
# HEADER (LOGOS)
# ================================
col1, col2 = st.columns([1, 6])

with col1:
    st.image("logo_PLUSE.QPredict.png", width=120)

with col2:
    st.image("logo_digitide.jpg", width=120)

st.title("🔍 System Outage Severity Prediction")

# ================================
# TABS
# ================================
tab1, tab2 = st.tabs(["📂 Batch Prediction", "⚡ Single Prediction"])

# =========================================================
# 📂 BATCH PREDICTION (WITH DELAY)
# =========================================================
with tab1:

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Uploaded Data")
        st.dataframe(df.head())

        try:
            st.info("📂 File uploaded successfully. Starting AI processing...")

            # 🔥 PROGRESS BAR (6 seconds total)
            progress_bar = st.progress(0)

            for i in range(100):
                time.sleep(0.06)   # 100 * 0.06 = ~6 seconds
                progress_bar.progress(i + 1)

            # 🔥 SPINNER
            with st.spinner("🤖 AI Model is analyzing outage patterns..."):

                # Drop unwanted columns
                drop_cols = ["Outage_ID", "End_Time"]
                df_model = df.drop(columns=[col for col in drop_cols if col in df.columns])

                # Same encoding as training
                df_model = pd.get_dummies(df_model)

                # Align with training features
                df_model = df_model.reindex(columns=feature_columns, fill_value=0)

                # Scale
                df_scaled = scaler.transform(df_model)

                # Predict
                preds = model.predict(df_scaled)
                pred_labels = label_encoder.inverse_transform(preds)

                # Confidence
                proba = model.predict_proba(df_scaled)
                confidence = np.max(proba, axis=1)

                # Add output
                df["Predicted_Severity"] = pred_labels
                df["Confidence"] = confidence

            # ✅ SUCCESS MESSAGE
            st.success(f"✅ Processing Completed! {len(df)} rows analyzed.")

            st.subheader("📊 Prediction Results")
            st.dataframe(df.head())

            # Download
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Results",
                csv,
                "predicted_output.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")

# =========================================================
# ⚡ SINGLE PREDICTION (UNCHANGED)
# =========================================================
with tab2:

    st.subheader("⚡ Predict Single Outage")

    col1, col2, col3 = st.columns(3)

    with col1:
        duration = st.slider("Duration (minutes)", 0, 600, 120)
        system = st.number_input("System Affected", 1000, 1100)

    with col2:
        resolution = st.slider("Resolution Time", 0, 600, 150)
        cause = st.selectbox("Cause", ["Network Issue", "Software Bug", "Hardware Failure", "Power Outage", "Unknown"])

    with col3:
        region = st.selectbox("Region", ["North", "South", "East", "West"])
        resolved_by = st.selectbox("Resolved By", ["Engineer", "Technician", "Vendor", "Support Team"])

    hour = st.slider("Hour", 0, 23)
    day = st.slider("Day of Week", 0, 6)
    month = st.slider("Month", 1, 12)

    if st.button("🚀 Predict Severity"):

        input_df = pd.DataFrame({
            "Duration (minutes)": [duration],
            "System_Affected": [system],
            "Resolution_Time (minutes)": [resolution],
            "Cause": [cause],
            "Region": [region],
            "Resolved_By": [resolved_by],
            "hour": [hour],
            "dayofweek": [day],
            "month": [month]
        })

        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)
        pred_label = label_encoder.inverse_transform(pred)[0]

        prob = model.predict_proba(input_scaled)
        confidence = np.max(prob)

        st.markdown("### 🔍 Prediction Result")

        if pred_label == "high":
            st.error(f"🚨 HIGH Severity")
        elif pred_label == "medium":
            st.warning(f"⚠️ MEDIUM Severity")
        elif pred_label == "low":
            st.info(f"🔹 LOW Severity")
        else:
            st.success(f"✅ NORMAL")

        st.metric("Confidence Score", f"{confidence:.2f}")

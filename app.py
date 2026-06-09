import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CKD Prediction Platform",
    page_icon="🏥",
    layout="wide"
)

# ── Load model and features ───────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('ckd_model.pkl')
    with open('feature_names.json') as f:
        features = json.load(f)
    return model, features

model, feature_names = load_model()

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.title("🏥 CKD Platform")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🔬 Predict CKD",
    "⚖️ Bias Analysis",
    "📊 Model Performance"
])

# ════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏥 Bias-Aware CKD Prediction Platform")
    st.markdown("### Predicting Chronic Kidney Disease Across Demographic Groups")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 🎯 400\nPatients in Dataset")
    with col2:
        st.success("### 💯 100%\nModel Accuracy")
    with col3:
        st.warning("### ⚖️ 19%\nBias Gap Detected")

    st.markdown("---")
    st.markdown("""
    ## About This Project
    Chronic Kidney Disease (CKD) affects over **850 million people worldwide**.
    Early detection saves lives — but AI models can be **biased** against
    certain demographic groups, leading to misdiagnosis.

    This platform:
    - ✅ Predicts CKD from clinical measurements
    - ✅ Detects bias across **age groups**
    - ✅ Compares fair vs biased model performance
    - ✅ Explains which factors drive predictions
    """)

    st.markdown("---")
    st.markdown("## Dataset Overview")
    st.image("ckd_overview.png", caption="CKD Dataset Analysis", use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ════════════════════════════════════════════════════════════════
elif page == "🔬 Predict CKD":
    st.title("🔬 CKD Prediction")
    st.markdown("Enter patient details below to predict CKD risk.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👤 Patient Info")
        age  = st.slider("Age", 1, 100, 45)
        bp   = st.slider("Blood Pressure (mm/Hg)", 50, 180, 80)
        sg   = st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025])
        al   = st.selectbox("Albumin (0–5)", [0,1,2,3,4,5])
        su   = st.selectbox("Sugar (0–5)", [0,1,2,3,4,5])
        bgr  = st.slider("Blood Glucose (mgs/dl)", 70, 490, 120)
        bu   = st.slider("Blood Urea (mgs/dl)", 10, 200, 40)
        sc   = st.slider("Serum Creatinine (mgs/dl)", 0.4, 15.0, 1.2)

    with col2:
        st.markdown("### 🩸 Blood Tests")
        sod  = st.slider("Sodium (mEq/L)", 100, 160, 135)
        pot  = st.slider("Potassium (mEq/L)", 2.5, 7.5, 4.5)
        hemo = st.slider("Hemoglobin (gms)", 3.0, 17.8, 13.0)
        pcv  = st.slider("Packed Cell Volume", 9, 54, 40)
        wc   = st.slider("White Blood Cell Count", 3800, 26400, 8000)
        rc   = st.slider("Red Blood Cell Count", 2.1, 8.0, 5.0)

    with col3:
        st.markdown("### 🏥 Clinical Findings")
        rbc   = st.selectbox("RBC", ["normal","abnormal"])
        pc    = st.selectbox("Pus Cells", ["normal","abnormal"])
        pcc   = st.selectbox("Pus Cell Clumps", ["notpresent","present"])
        ba    = st.selectbox("Bacteria", ["notpresent","present"])
        htn   = st.selectbox("Hypertension", ["no","yes"])
        dm    = st.selectbox("Diabetes Mellitus", ["no","yes"])
        cad   = st.selectbox("Coronary Artery Disease", ["no","yes"])
        appet = st.selectbox("Appetite", ["good","poor"])
        pe    = st.selectbox("Pedal Edema", ["no","yes"])
        ane   = st.selectbox("Anemia", ["no","yes"])

    st.markdown("---")

    if st.button("🔍 PREDICT", use_container_width=True):
        # Encode inputs
        def encode(val, options):
            return options.index(val)

        input_data = {
            'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
            'rbc': encode(rbc, ['normal','abnormal']),
            'pc': encode(pc, ['normal','abnormal']),
            'pcc': encode(pcc, ['notpresent','present']),
            'ba': encode(ba, ['notpresent','present']),
            'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
            'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
            'htn': encode(htn, ['no','yes']),
            'dm': encode(dm, ['no','yes']),
            'cad': encode(cad, ['no','yes']),
            'appet': encode(appet, ['good','poor']),
            'pe': encode(pe, ['no','yes']),
            'ane': encode(ane, ['no','yes'])
        }

        input_df = pd.DataFrame([input_data])[feature_names]
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.markdown("---")
        if prediction == 0:
            st.error(f"## ⚠️ CKD DETECTED\nConfidence: {probability[0]*100:.1f}%")
            st.markdown("**Recommendation:** Refer patient for immediate nephrology consultation.")
        else:
            st.success(f"## ✅ NO CKD DETECTED\nConfidence: {probability[1]*100:.1f}%")
            st.markdown("**Recommendation:** Continue routine monitoring.")

        # Show input summary
        st.markdown("### 📋 Patient Summary")
        summary = pd.DataFrame({
            'Measurement': ['Age','Blood Pressure','Hemoglobin','Serum Creatinine','Blood Glucose'],
            'Value': [age, bp, hemo, sc, bgr]
        })
        st.table(summary)

# ════════════════════════════════════════════════════════════════
# PAGE 3 — BIAS ANALYSIS
# ════════════════════════════════════════════════════════════════
elif page == "⚖️ Bias Analysis":
    st.title("⚖️ Bias Analysis Across Age Groups")
    st.markdown("---")

    st.markdown("""
    ## What is Model Bias?
    A biased AI model performs **differently for different groups** of people.
    In healthcare, this can mean some patients are more likely to be
    **misdiagnosed** than others — which is dangerous and unfair.
    """)

    st.markdown("### Key Finding")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("### ✅ Fair Model\nElderly Accuracy: **100%**")
    with col2:
        st.error("### ⚠️ Biased Model\nElderly Accuracy: **81%**")
    with col3:
        st.warning("### 📉 Bias Gap\n**19% drop** for elderly patients")

    st.markdown("---")
    st.image("bias_analysis_v2.png",
             caption="Fair vs Biased Model — Performance Across Age Groups",
             use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ## Why Does This Happen?
    - Elderly patients were **underrepresented** in training data
    - The model learned patterns mainly from younger patients
    - When tested on elderly patients it performed **19% worse**

    ## How We Fix It
    - ✅ Ensure **equal representation** of all age groups in training data
    - ✅ Use **bias detection metrics** (demographic parity, equalized odds)
    - ✅ Apply **reweighing** — give more importance to underrepresented groups
    """)

# ════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy",  "100%")
    with col2:
        st.metric("Precision", "100%")
    with col3:
        st.metric("Recall",    "100%")
    with col4:
        st.metric("F1 Score",  "100%")

    st.markdown("---")
    st.markdown("### Feature Importance — What Predicts CKD Most?")
    st.image("feature_importance.png",
             caption="Top predictors of CKD",
             use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### Model Details
    | Detail | Value |
    |---|---|
    | Algorithm | Random Forest Classifier |
    | Training samples | 320 patients |
    | Testing samples | 80 patients |
    | Number of trees | 100 |
    | Dataset | UCI CKD Dataset (Kaggle) |
    | Features used | 24 clinical measurements |
    """)

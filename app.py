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

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F0F4F8; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0A2342;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-size: 16px;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] p {
        color: white !important;
    }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #02C39A;
    }
    .metric-card h2 { color: #065A82; margin: 0; font-size: 2.2em; }
    .metric-card p  { color: #64748B; margin: 0; font-size: 0.95em; }

    /* How to use box */
    .how-to-box {
        background: #EFF6FF;
        border-radius: 12px;
        padding: 18px 22px;
        border-left: 5px solid #065A82;
        margin-bottom: 18px;
    }

    /* Result boxes */
    .result-ckd {
        background: #FEE2E2;
        border-radius: 12px;
        padding: 22px;
        border-left: 6px solid #E63946;
        text-align: center;
    }
    .result-no-ckd {
        background: #DCFCE7;
        border-radius: 12px;
        padding: 22px;
        border-left: 6px solid #2DC653;
        text-align: center;
    }

    /* Section headers */
    .section-header {
        background: #0A2342;
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 1.1em;
        font-weight: bold;
    }

    /* Key finding banner */
    .finding-banner {
        background: #0A2342;
        color: #02C39A;
        padding: 14px 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.05em;
        font-weight: bold;
        margin: 15px 0;
    }

    /* Predict button */
    .stButton > button {
        background-color: #065A82;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        width: 100%;
        transition: background 0.3s;
    }
    .stButton > button:hover {
        background-color: #02C39A;
        color: #0A2342;
    }

    /* Hide default header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('ckd_model.pkl')
    with open('feature_names.json') as f:
        features = json.load(f)
    return model, features

model, feature_names = load_model()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("# 🏥 CKD Platform")
st.sidebar.markdown("*Bias-Aware Prediction System*")
st.sidebar.markdown("---")
page = st.sidebar.radio("", [
    "🏠 Home",
    "🔬 Predict CKD",
    "⚖️ Bias Analysis",
    "📊 Model Performance"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** UCI CKD Dataset")
st.sidebar.markdown("**Model:** Random Forest")
st.sidebar.markdown("**Accuracy:** 100%")
st.sidebar.markdown("**Patients:** 400")

# ════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    # Hero section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0A2342 0%, #065A82 100%); 
                padding: 40px 30px; border-radius: 16px; margin-bottom: 25px;'>
        <h1 style='color: white; margin:0; font-size: 2.1em;'>
            🏥 Bias-Aware CKD Prediction Platform
        </h1>
        <p style='color: #02C39A; font-size: 1.15em; margin: 8px 0 0 0;'>
            Predicting Chronic Kidney Disease Fairly Across Demographic Groups
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='metric-card'>
            <h2>400</h2><p>Patients in Dataset</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card' style='border-left-color:#2DC653'>
            <h2 style='color:#2DC653'>100%</h2><p>Model Accuracy</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card' style='border-left-color:#E63946'>
            <h2 style='color:#E63946'>19%</h2><p>Bias Gap Detected</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # About section
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📌 About This Project")
        st.markdown("""
        Chronic Kidney Disease affects over **850 million people worldwide**.
        Early detection saves lives — but AI models can be **biased** against
        certain demographic groups, leading to misdiagnosis.

        This platform:
        - ✅ Predicts CKD from clinical measurements
        - ✅ Detects bias across **age groups**
        - ✅ Compares fair vs biased model performance
        - ✅ Explains which factors drive predictions
        """)
    with col2:
        st.markdown("### 🔑 Key Finding")
        st.markdown("""<div class='finding-banner'>
            When elderly patients are underrepresented in training data,
            model accuracy drops by <span style='color:white; font-size:1.3em'>19%</span>
            for that group
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        This proves why **bias-aware AI** is critical in healthcare.
        A model that performs worse for elderly patients could lead to
        missed diagnoses and unequal treatment.
        """)

    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    st.image("ckd_overview.png", caption="CKD Dataset Analysis", use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ════════════════════════════════════════════════════════════════
elif page == "🔬 Predict CKD":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0A2342 0%, #065A82 100%);
                padding: 25px 30px; border-radius: 16px; margin-bottom: 20px;'>
        <h2 style='color:white; margin:0;'>🔬 CKD Prediction Tool</h2>
        <p style='color:#02C39A; margin:5px 0 0 0;'>
            Enter patient clinical measurements to get an instant CKD prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    # How to use guide
    st.markdown("""
    <div class='how-to-box'>
        <h4 style='color:#065A82; margin-top:0;'>📖 How to Use This Tool</h4>
        <ol style='color:#1E293B; margin:0; padding-left:18px;'>
            <li>Fill in the patient's clinical measurements in the three columns below</li>
            <li>Use the sliders for numerical values and dropdowns for categorical values</li>
            <li>Leave unknown values at their default — the model handles missing data</li>
            <li>Click the <strong>PREDICT</strong> button at the bottom to get the result</li>
            <li>The result shows CKD status with a confidence percentage</li>
        </ol>
        <p style='color:#64748B; margin:10px 0 0 0; font-size:0.9em;'>
            ⚠️ This tool is for research purposes only and should not replace professional medical diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Input form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='section-header'>👤 Patient Information</div>", unsafe_allow_html=True)
        age  = st.slider("Age (years)", 1, 100, 45)
        bp   = st.slider("Blood Pressure (mm/Hg)", 50, 180, 80)
        sg   = st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025])
        al   = st.selectbox("Albumin Level (0–5)", [0,1,2,3,4,5])
        su   = st.selectbox("Sugar Level (0–5)", [0,1,2,3,4,5])
        bgr  = st.slider("Blood Glucose (mgs/dl)", 70, 490, 120)
        bu   = st.slider("Blood Urea (mgs/dl)", 10, 200, 40)
        sc   = st.slider("Serum Creatinine (mgs/dl)", 0.4, 15.0, 1.2)

    with col2:
        st.markdown("<div class='section-header'>🩸 Blood Test Results</div>", unsafe_allow_html=True)
        sod  = st.slider("Sodium (mEq/L)", 100, 160, 135)
        pot  = st.slider("Potassium (mEq/L)", 2.5, 7.5, 4.5)
        hemo = st.slider("Hemoglobin (gms)", 3.0, 17.8, 13.0)
        pcv  = st.slider("Packed Cell Volume", 9, 54, 40)
        wc   = st.slider("White Blood Cell Count", 3800, 26400, 8000)
        rc   = st.slider("Red Blood Cell Count", 2.1, 8.0, 5.0)

    with col3:
        st.markdown("<div class='section-header'>🏥 Clinical Findings</div>", unsafe_allow_html=True)
        rbc   = st.selectbox("Red Blood Cells", ["normal","abnormal"])
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
    if st.button("🔍 PREDICT CKD", use_container_width=True):
        def encode(val, options): return options.index(val)

        input_data = {
            'age':age,'bp':bp,'sg':sg,'al':al,'su':su,
            'rbc':encode(rbc,['normal','abnormal']),
            'pc':encode(pc,['normal','abnormal']),
            'pcc':encode(pcc,['notpresent','present']),
            'ba':encode(ba,['notpresent','present']),
            'bgr':bgr,'bu':bu,'sc':sc,'sod':sod,'pot':pot,
            'hemo':hemo,'pcv':pcv,'wc':wc,'rc':rc,
            'htn':encode(htn,['no','yes']),
            'dm':encode(dm,['no','yes']),
            'cad':encode(cad,['no','yes']),
            'appet':encode(appet,['good','poor']),
            'pe':encode(pe,['no','yes']),
            'ane':encode(ane,['no','yes'])
        }

        input_df = pd.DataFrame([input_data])[feature_names]
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.markdown("---")
        st.markdown("### 🧾 Prediction Result")

        col_r1, col_r2 = st.columns([2,1])
        with col_r1:
            if prediction == 0:
                conf = probability[0]*100
                st.markdown(f"""
                <div class='result-ckd'>
                    <h2 style='color:#E63946; margin:0;'>⚠️ CKD DETECTED</h2>
                    <h3 style='color:#E63946; margin:5px 0;'>Confidence: {conf:.1f}%</h3>
                    <p style='color:#1E293B; margin:0;'>
                        <strong>Recommendation:</strong> Refer patient for immediate nephrology consultation.
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                conf = probability[1]*100
                st.markdown(f"""
                <div class='result-no-ckd'>
                    <h2 style='color:#2DC653; margin:0;'>✅ NO CKD DETECTED</h2>
                    <h3 style='color:#2DC653; margin:5px 0;'>Confidence: {conf:.1f}%</h3>
                    <p style='color:#1E293B; margin:0;'>
                        <strong>Recommendation:</strong> Continue routine health monitoring.
                    </p>
                </div>""", unsafe_allow_html=True)

        with col_r2:
            st.markdown("### 📋 Key Values")
            summary = pd.DataFrame({
                'Measurement': ['Age','Blood Pressure','Hemoglobin','Serum Creatinine','Blood Glucose'],
                'Value': [f"{age} yrs", f"{bp} mmHg", f"{hemo} g", f"{sc} mg/dl", f"{bgr} mg/dl"]
            })
            st.table(summary)

# ════════════════════════════════════════════════════════════════
# PAGE 3 — BIAS ANALYSIS
# ════════════════════════════════════════════════════════════════
elif page == "⚖️ Bias Analysis":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #E63946 0%, #0A2342 100%);
                padding: 25px 30px; border-radius: 16px; margin-bottom: 20px;'>
        <h2 style='color:white; margin:0;'>⚖️ Bias Analysis Across Age Groups</h2>
        <p style='color:#02C39A; margin:5px 0 0 0;'>
            Comparing fair model vs biased model performance
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What is Model Bias?")
    st.markdown("""
    A biased AI model performs **differently for different groups** of people.
    In healthcare, this means some patients are more likely to be **misdiagnosed** — which is dangerous and unfair.
    """)

    st.markdown("""<div class='finding-banner'>
        Key Finding: When elderly patients are underrepresented in training data,
        model accuracy drops by 19% for that group
    </div>""", unsafe_allow_html=True)

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
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ❓ Why Does This Happen?")
        st.markdown("""
        - Elderly patients were **underrepresented** in training data
        - The model learned patterns mainly from younger patients
        - When tested on elderly patients it performed **19% worse**
        - This is a common problem in real-world medical datasets
        """)
    with col2:
        st.markdown("### ✅ How We Fix It")
        st.markdown("""
        - Ensure **equal representation** of all age groups in training
        - Use **bias detection metrics** like demographic parity
        - Apply **reweighing** — give more importance to underrepresented groups
        - Continuously **monitor** model performance across groups after deployment
        """)

# ════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0A2342 0%, #02C39A 100%);
                padding: 25px 30px; border-radius: 16px; margin-bottom: 20px;'>
        <h2 style='color:white; margin:0;'>📊 Model Performance</h2>
        <p style='color:white; margin:5px 0 0 0; opacity:0.85;'>
            Random Forest Classifier — Full Evaluation Results
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class='metric-card'>
            <h2>100%</h2><p>Accuracy</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card' style='border-left-color:#065A82'>
            <h2 style='color:#065A82'>100%</h2><p>Precision</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card' style='border-left-color:#2DC653'>
            <h2 style='color:#2DC653'>100%</h2><p>Recall</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class='metric-card' style='border-left-color:#F4A261'>
            <h2 style='color:#F4A261'>100%</h2><p>F1 Score</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Feature Importance — What Predicts CKD Most?")
    st.image("feature_importance.png",
             caption="Top predictors of CKD — Hemoglobin and Serum Creatinine are most important",
             use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚙️ Model Details")
        st.markdown("""
        | Detail | Value |
        |---|---|
        | Algorithm | Random Forest |
        | Number of Trees | 100 |
        | Training Samples | 320 (80%) |
        | Testing Samples | 80 (20%) |
        | Features Used | 24 clinical measurements |
        | Dataset | UCI CKD (via Kaggle) |
        """)
    with col2:
        st.markdown("### 🏆 Top 5 Predictors")
        st.markdown("""
        | Rank | Feature | Importance |
        |---|---|---|
        | 1 | Hemoglobin | 16.7% |
        | 2 | Packed Cell Volume | 16.6% |
        | 3 | Serum Creatinine | 15.5% |
        | 4 | Specific Gravity | 11.1% |
        | 5 | Red Blood Cells | 9.3% |
        """)

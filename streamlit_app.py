import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ─── Load Model & Scaler ───────────────────────────────────────
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="centered"
)

# ─── Title ────────────────────────────────────────────────────
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the patient details below to predict diabetes risk.")
st.divider()

# ─── Input Form ───────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=110)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 
                                         min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=25)

st.divider()

# ─── Predict Button ───────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):

    # Prepare input
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure,
                                 skin_thickness, insulin, bmi,
                                 diabetes_pedigree, age]],
                               columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                        'SkinThickness', 'Insulin', 'BMI',
                                        'DiabetesPedigreeFunction', 'Age'])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    diabetes_prob = probability[0][1] * 100
    healthy_prob  = probability[0][0] * 100

    st.divider()

    # ─── Result ───────────────────────────────────────────────
    if prediction[0] == 1:
        st.error(f"### 🔴 Result: Diabetic")
        st.metric("Diabetes Probability", f"{diabetes_prob:.2f}%")
    else:
        st.success(f"### 🟢 Result: Not Diabetic")
        st.metric("Healthy Probability", f"{healthy_prob:.2f}%")

    # ─── Probability Bar ──────────────────────────────────────
    st.divider()
    st.subheader("📊 Prediction Probabilities")

    prob_df = pd.DataFrame({
        'Condition': ['Not Diabetic 🟢', 'Diabetic 🔴'],
        'Probability %': [round(healthy_prob, 2), round(diabetes_prob, 2)]
    })
    st.bar_chart(prob_df.set_index('Condition'))

    # ─── Input Summary ────────────────────────────────────────
    st.divider()
    st.subheader("📋 Patient Input Summary")
    st.dataframe(input_data, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("⚠️ This app is for educational purposes only. Consult a doctor for medical advice.")
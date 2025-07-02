import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")
st.title("🩺 Diabetes Risk Prediction")
st.write("Please enter patient information:")

# ========== إدخال البيانات ==========
age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)

family_history = st.radio("Family history of diabetes?", ("Yes", "No"))
smoking_status = st.radio("Is the patient a smoker?", ("Yes", "No"))
physical_activity = st.slider("Physical Activity (hours per week)", 0, 20, 3)

# ========== زر التنبؤ ==========
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Age": age,
        "BMI": bmi,
        "Blood Pressure": blood_pressure,
        "Family History": family_history,
        "Smoking Status": smoking_status,
        "Physical Activity (hours/week)": physical_activity
    }])

    # تحميل النموذج والمعالج
    model = joblib.load("final_Dia_model.pkl")
    preprocessor = joblib.load("dia_preprocessor.pkl")

    # تحويل البيانات
    processed_data = preprocessor.transform(input_data)

    # التنبؤ
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1] if hasattr(model, "predict_proba") else None

    # عرض النتيجة
    st.subheader("🧪 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ The model predicts that the patient **is at risk** of diabetes.")
    else:
        st.success("✅ The model predicts that the patient **is NOT at risk** of diabetes.")

    if probability is not None:
        st.write(f"**Probability of having diabetes:** {probability:.2%}")

st.caption("Developed by Ali Ahmed Zaki ·")

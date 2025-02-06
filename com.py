import streamlit as st
import numpy as np
import pandas as pd
import joblib
import h5py
import io

# Function to Load Model & Scaler from .h5
def load_model_from_h5(filename, model_key, scaler_key):
    with h5py.File(filename, "r") as h5f:
        model_bytes = io.BytesIO(h5f[model_key][()])
        scaler_bytes = io.BytesIO(h5f[scaler_key][()])
    model = joblib.load(model_bytes)
    scaler = joblib.load(scaler_bytes)
    return model, scaler

# Load Models and Scalers
xgb_model, xgb_scaler = load_model_from_h5("./xgb_model.h5", "xgb_model", "scaler")
rf_model, rf_scaler = load_model_from_h5("./rf_model.h5", "rf_model", "scaler")

# Streamlit UI Navigation
st.sidebar.title("Navigation")
model_choice = st.sidebar.radio("Choose Model", ["XGBoost", "Random Forest"])

st.title("Cardiovascular Disease Risk Prediction")
st.write("Enter the patient details below to estimate the probability of cardiovascular disease.")

if model_choice == "XGBoost":
    # Inputs for XGBoost Model
    age = st.number_input("Age", min_value=18, max_value=120, value=60)
    gender = st.radio("Gender", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (FBS)", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=140)
    exang = st.radio("Exercise Induced Angina (exang)", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=6.0, value=1.0)
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
    
    # Convert categorical inputs to numerical values
    gender = 1 if gender == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    input_data = {
        'age': age, 'gender': gender, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'ca': ca
    }
    model, scaler = xgb_model, xgb_scaler

else:
    # Inputs for Random Forest Model
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    chestpain = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
    restingBP = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    serumcholestrol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, value=200)
    fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], index=0)
    restingrelectro = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=1)
    maxheartrate = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exerciseangia = st.selectbox("Exercise-Induced Angina", ["No", "Yes"], index=0)
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    noofmajorvessels = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=1)
    
    # Convert categorical inputs
    gender = 1 if gender == "Male" else 0
    fastingbloodsugar = 1 if fastingbloodsugar == "Yes" else 0
    exerciseangia = 1 if exerciseangia == "Yes" else 0

    input_data = {
        'age': age, 'gender': gender, 'chestpain': chestpain, 'restingBP': restingBP,
        'serumcholestrol': serumcholestrol, 'fastingbloodsugar': fastingbloodsugar,
        'restingrelectro': restingrelectro, 'maxheartrate': maxheartrate,
        'exerciseangia': exerciseangia, 'oldpeak': oldpeak, 'noofmajorvessels': noofmajorvessels
    }
    model, scaler = rf_model, rf_scaler

# Function to Predict CVD Risk
def predict_cvd(model, scaler, input_data):
    input_scaled = scaler.transform(pd.DataFrame([input_data]))
    probability = model.predict_proba(input_scaled)[0][1] * 100
    return probability

# Predict Button
if st.button("Predict"):
    probability = predict_cvd(model, scaler, input_data)
    st.success(f"💡 Probability of Cardiovascular Disease: {probability:.2f}%")
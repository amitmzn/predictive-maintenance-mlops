import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Update 'filename' below if you saved your model with a different name in your notebook
model_path = hf_hub_download(repo_id="amitmzn/predictive-maintenance-model", filename="best_model.joblib")
model = joblib.load(model_path)

st.title("Predictive Maintenance - Engine Condition Prediction")
st.write("Predict whether an engine requires maintenance based on real-time sensor readings.")

# Input fields using defaults based on the dataset means you explored earlier
engine_rpm = st.number_input("Engine RPM", 0, 5000, 791)
lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", 0.0, 20.0, 3.3)
fuel_pressure = st.number_input("Fuel Pressure (bar)", 0.0, 50.0, 6.6)
coolant_pressure = st.number_input("Coolant Pressure (bar)", 0.0, 20.0, 2.3)
lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", 0.0, 150.0, 77.6)
coolant_temp = st.number_input("Coolant Temperature (°C)", 0.0, 250.0, 78.4)

# Constructing the dataframe with EXACT column names matching your training dataset
input_data = pd.DataFrame([[
    engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp
]], columns=['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp'])

if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.error(f"Engine Faulty / Needs Maintenance! (Probability: {probability[1]:.2%})")
    else:
        st.success(f"Engine is Normal / Healthy. (Probability: {probability[0]:.2%})")

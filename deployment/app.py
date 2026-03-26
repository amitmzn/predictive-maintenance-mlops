import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import numpy as np

def add_features(df):
    df = df.copy()
    df['RPM_per_Oil_Pressure'] = df['Engine_RPM'] / (df['Lub_Oil_Pressure'] + 0.01)
    df['RPM_per_Fuel_Pressure'] = df['Engine_RPM'] / (df['Fuel_Pressure'] + 0.01)
    df['RPM_per_Coolant_Pressure'] = df['Engine_RPM'] / (df['Coolant_Pressure'] + 0.01)
    df['RPM_per_Oil_Temp'] = df['Engine_RPM'] / (df['Lub_Oil_Temperature'] + 0.01)
    df['RPM_per_Coolant_Temp'] = df['Engine_RPM'] / (df['Coolant_Temperature'] + 0.01)

    df['Engine_Load_Fuel'] = df['Engine_RPM'] * df['Fuel_Pressure']
    df['Engine_Load_Oil'] = df['Engine_RPM'] * df['Lub_Oil_Pressure']
    df['Engine_Load_Coolant'] = df['Engine_RPM'] * df['Coolant_Pressure']

    df['Oil_Temp_Diff'] = df['Lub_Oil_Temperature'] - df['Coolant_Temperature']
    df['Oil_Coolant_Pressure_Ratio'] = df['Lub_Oil_Pressure'] / (df['Coolant_Pressure'] + 0.01)
    df['RPM_low'] = (df['Engine_RPM'] < 600).astype(int)
    df['RPM_high'] = (df['Engine_RPM'] > 900).astype(int)
    df['LubTemp_low'] = (df['Lub_Oil_Temperature'] < 76).astype(int)

    df['RPM_sq'] = df['Engine_RPM'] ** 2
    df['RPM_log'] = np.log1p(df['Engine_RPM'])

    return df

model_path = hf_hub_download(repo_id="amitmzn/predictive-maintenance-model", filename="best_maintenance_model.joblib")
model = joblib.load(model_path)

scaler_path = hf_hub_download(repo_id="amitmzn/predictive-maintenance-model", filename="scaler.joblib")
scaler = joblib.load(scaler_path)

imputer_path = hf_hub_download(repo_id="amitmzn/predictive-maintenance-model", filename="imputer.joblib")
imputer = joblib.load(imputer_path)

bounds_path = hf_hub_download(repo_id="amitmzn/predictive-maintenance-model", filename="preprocessing_bounds.joblib")
preprocess_bounds = joblib.load(bounds_path)
OPTIMAL_THRESHOLD = preprocess_bounds['optimal_threshold']
COOLANT_UPPER_BOUND = preprocess_bounds['coolant_upper_bound']

st.title("Predictive Maintenance - Engine Condition Prediction")
st.write("Predict whether an engine requires maintenance based on real-time sensor readings.")

engine_rpm = st.number_input("Engine RPM", 0, 5000, 791)
lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", 0.0, 20.0, 3.3)
fuel_pressure = st.number_input("Fuel Pressure (bar)", 0.0, 50.0, 6.6)
coolant_pressure = st.number_input("Coolant Pressure (bar)", 0.0, 20.0, 2.3)
lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", 0.0, 150.0, 77.6)
coolant_temp = st.number_input("Coolant Temperature (°C)", 0.0, 250.0, 78.4)

if st.button("Predict Engine Condition"):
    # 1. Construct the dataframe
    input_data = pd.DataFrame([[
        engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp
    ]], columns=['Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure', 'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature'])

    # 2. Outlier capping (same as training)
    input_data.loc[input_data['Coolant_Temperature'] > COOLANT_UPPER_BOUND, 'Coolant_Temperature'] = np.nan

    # 3. Impute missing values (MANDATORY - same as training)
    input_data = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)

    # 4. Feature engineering (same as training)
    processed_data = add_features(input_data)

    # 5. Scale features (MANDATORY - same as training)
    scaled_array = scaler.transform(processed_data)
    processed_data = pd.DataFrame(scaled_array, columns=processed_data.columns)

    # 6. Align columns with model
    if hasattr(model, 'feature_names_in_'):
        processed_data = processed_data[model.feature_names_in_]

    # 7. Predict using optimal threshold 
    probability = model.predict_proba(processed_data)[0]
    faulty_prob = probability[1]
    prediction = 1 if faulty_prob >= OPTIMAL_THRESHOLD else 0

    if prediction == 1:
        st.error(f"Engine Faulty / Needs Maintenance! (Probability: {faulty_prob:.2%})")
    else:
        st.success(f"Engine is Normal / Healthy. (Probability: {probability[0]:.2%})")

    st.info(f"Decision threshold: {OPTIMAL_THRESHOLD:.2%}")

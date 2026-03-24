import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# 1. Load Data
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/amitmzn/Predictive-Maintenance/engine_data.csv"

df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape}")

df = df.drop([col for col in df.columns if 'Unnamed' in col], axis=1, errors='ignore')
df.columns = [col.strip().replace(' ', '_') for col in df.columns]

column_mapping = {
    'Engine_rpm': 'Engine_RPM',
    'Lub_oil_pressure': 'Lub_Oil_Pressure',
    'Fuel_pressure': 'Fuel_Pressure',
    'Coolant_pressure': 'Coolant_Pressure',
    'lub_oil_temp': 'Lub_Oil_Temperature',
    'Coolant_temp': 'Coolant_Temperature',
}
df.rename(columns=column_mapping, inplace=True)

X = df.drop('Engine_Condition', axis=1)
y = df['Engine_Condition']

# Handle missing values
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# 2. Strict Train / Validation / Test Split (Zero Data Leakage)
X_train_full, X_test_raw, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

# 3. Save Raw Splits Locally
data_dir = 'predictive_maintenance_project/data'
os.makedirs(data_dir, exist_ok=True)

X_train_raw.to_csv("Xtrain.csv", index=False)
X_val_raw.to_csv("Xval.csv", index=False)
X_test_raw.to_csv("Xtest.csv", index=False)

y_train.to_csv("ytrain.csv", index=False)
y_val.to_csv("yval.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

# Upload to Hugging Face
files = ["Xtrain.csv", "Xval.csv","Xtest.csv", "ytrain.csv", "yval.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="amitmzn/Predictive-Maintenance",
        repo_type="dataset",
    )

print("Data preparation complete!")

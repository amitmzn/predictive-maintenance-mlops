
import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import randint, uniform
from huggingface_hub import HfApi

RANDOM_STATE = 42
MODEL_REPO = "amitmzn/predictive-maintenance-model"

# 1. Finalized Feature Engineering
def add_features(dataframe):
    df = dataframe.copy()
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

if __name__ == "__main__":
    print("1. Loading Data...")
    data_path = "hf://datasets/amitmzn/Predictive-Maintenance/engine_data.csv"
    df = pd.read_csv(data_path)

    # Standardize column names
    df = df.drop([col for col in df.columns if 'Unnamed' in col], axis=1, errors='ignore')
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    column_mapping = {
        'Engine_rpm': 'Engine_RPM', 'Lub_oil_pressure': 'Lub_Oil_Pressure',
        'Fuel_pressure': 'Fuel_Pressure', 'Coolant_pressure': 'Coolant_Pressure',
        'lub_oil_temp': 'Lub_Oil_Temperature', 'Coolant_temp': 'Coolant_Temperature',
    }
    df.rename(columns=column_mapping, inplace=True)

    X = df.drop('Engine_Condition', axis=1)
    y = df['Engine_Condition']

    print("2. Train / Validation / Test Split...")
    X_train_full, X_test_raw, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train_full
    )

    print("3. Preprocessing (Learned entirely from Train split to prevent leakage)...")

    # Learn bounds from Train only
    q1 = X_train_raw['Coolant_Temperature'].quantile(0.25)
    q3 = X_train_raw['Coolant_Temperature'].quantile(0.75)
    coolant_upper_bound = q3 + 1.5 * (q3 - q1)

    # Cap outliers
    for frame in (X_train_raw, X_val_raw, X_test_raw):
        frame.loc[frame['Coolant_Temperature'] > coolant_upper_bound, 'Coolant_Temperature'] = np.nan

    # Fit Imputer on Train only
    imputer = SimpleImputer(strategy='median')
    train_imputed = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
    val_imputed = pd.DataFrame(imputer.transform(X_val_raw), columns=X_val_raw.columns, index=X_val_raw.index)

    # Feature Engineering
    train_feat = add_features(train_imputed)
    val_feat = add_features(val_imputed)

    # Fit Scaler on Train only
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(train_feat), columns=train_feat.columns, index=train_feat.index)
    X_val_scaled = pd.DataFrame(scaler.transform(val_feat), columns=val_feat.columns, index=val_feat.index)

    print("4. Training XGBoost with Randomized Search...")
    sample_weight_train = compute_sample_weight(class_weight='balanced', y=y_train)

    param_dist = {
        'n_estimators': randint(100, 400),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.15),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0.5, 2),
    }

    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', tree_method='hist'),
        param_distributions=param_dist,
        n_iter=30,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    xgb_search.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)

    best_xgb = XGBClassifier(
        **xgb_search.best_params_,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        tree_method='hist',
        early_stopping_rounds=30,
    )

    best_xgb.fit(
        X_train_scaled, y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )

    # Threshold Tuning on Validation set
    xgb_val_prob = best_xgb.predict_proba(X_val_scaled)[:, 1]
    val_precisions, val_recalls, val_thresholds = precision_recall_curve(y_val, xgb_val_prob)
    target_recall = 0.90
    valid_idx = np.where(val_recalls[:-1] >= target_recall)[0]
    optimal_threshold = float(val_thresholds[valid_idx[np.argmax(val_precisions[:-1][valid_idx])]]) if len(valid_idx) > 0 else 0.50

    print(f"Training Complete. Validation AUC: {roc_auc_score(y_val, xgb_val_prob):.4f}")
    print(f"Learned Optimal Threshold for 90% Recall: {optimal_threshold:.4f}")

    print("5. Saving Model Artifacts...")
    model_dir = "predictive_maintenance_project/model_building"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_xgb, os.path.join(model_dir, "best_maintenance_model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(imputer, os.path.join(model_dir, "imputer.joblib"))

    preprocess_artifacts = {
        'coolant_upper_bound': float(coolant_upper_bound),
        'optimal_threshold': optimal_threshold,
        'feature_columns': list(train_feat.columns),
    }
    joblib.dump(preprocess_artifacts, os.path.join(model_dir, "preprocessing_bounds.joblib"))

    print("6. Uploading All Artifacts to Hugging Face Model Hub...")
    api = HfApi(token=os.getenv("HF_TOKEN"))

    artifacts_to_upload = [
        "best_maintenance_model.joblib",
        "scaler.joblib",
        "imputer.joblib",
        "preprocessing_bounds.joblib",
    ]

    for filename in artifacts_to_upload:
        filepath = os.path.join(model_dir, filename)
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        print(f"  Uploaded: {filename}")

    print(f"All artifacts uploaded to {MODEL_REPO}")
    print("All artifacts successfully saved! Ready for UI deployment.")

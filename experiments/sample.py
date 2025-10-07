# emi_full_pipeline_allinone.py
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")

# ================== 1️⃣ Data Loading & Cleaning ==================
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop_duplicates()
    # Categories
    edu_categories = ['High School', 'Graduate', 'Post Graduate', 'Professional']
    company_categories = ['Startup', 'Small', 'Mid-size', 'Large Indian', 'MNC']
    house_type_categories = ["Rented", "Family", "Own"]
    emi_eligibility_categories = ['Not_Eligible', 'High_Risk', 'Eligible']
    # Cleaning numeric columns
    data['credit_score'] = pd.to_numeric(data['credit_score'], errors='coerce')
    data = data[(data['credit_score'] <= 900) & data['credit_score'].notna()].reset_index(drop=True)
    data = data[~((data["house_type"] == "Rented") & data["monthly_rent"].isna())].reset_index(drop=True)
    data['age'] = (data['age'].astype(str).str.strip().str.replace(r'[^0-9.]', '', regex=True)
                   .str.replace(r'(\.\d*)\..*', r'\1', regex=True).astype(float))
    data['gender'] = data['gender'].astype(str).str.strip().str.lower()
    data['gender'] = data['gender'].replace({'^m$': 'male', '^f$': 'female'}, regex=True)
    data['gender'] = data['gender'].replace(r'^\s*$', pd.NA, regex=True).map({'male': 0, 'female': 1})
    data['marital_status'] = data['marital_status'].astype(str).str.strip().str.lower()
    data['marital_status'] = data['marital_status'].replace(r'^\s*$', pd.NA, regex=True).map({'single':0,'married':1})
    data["education"] = pd.Series(pd.Categorical(data["education"], categories=edu_categories, ordered=True).codes).replace({-1: pd.NA})
    data["monthly_salary"] = (data['monthly_salary'].astype(str).str.strip().str.replace(r'[^0-9.]','',regex=True)
                              .str.replace(r'(\.\d*)\..*', r'\1', regex=True).astype(float))
    data = pd.get_dummies(data, columns=["employment_type"], dtype="int")
    data['company_type'] = pd.Series(pd.Categorical(data["company_type"], categories=company_categories, ordered=True).codes).replace({-1: pd.NA})
    data['house_type'] = pd.Series(pd.Categorical(data["house_type"], categories=house_type_categories, ordered=True).codes).replace({-1: pd.NA})
    data["bank_balance"] = pd.to_numeric(data['bank_balance'].astype(str).str.strip()
                                        .str.replace(r'[^0-9.]','',regex=True).str.replace(r'(\.\d*)\..*',r'\1',regex=True),
                                        errors="coerce")
    data['existing_loans'] = data['existing_loans'].astype(str).str.strip().str.lower()
    data['existing_loans'] = data['existing_loans'].replace(r'^\s*$', pd.NA, regex=True).map({'no':0,'yes':1})
    data["emi_eligibility"] = pd.Series(pd.Categorical(data["emi_eligibility"], categories=emi_eligibility_categories, ordered=True).codes).replace({-1: pd.NA})
    data = pd.get_dummies(data, columns=["emi_scenario"], dtype="int")
    data = data.dropna()
    return data

# ================== 2️⃣ Feature Engineering ==================
def feature_engineering(data):
    data['debt_to_income'] = data['current_emi_amount'] / data['monthly_salary']
    data['expense_to_income'] = (data['school_fees'] + data['college_fees'] + data['travel_expenses'] + 
                                 data['groceries_utilities'] + data['other_monthly_expenses']) / data['monthly_salary']
    data['affordability_ratio'] = (data['monthly_salary'] - data['current_emi_amount'] - 
                                   data['monthly_rent'] - data['other_monthly_expenses']) / data['requested_amount']
    return data

# ================== 3️⃣ Detect Features ==================
def detect_features(data, target_cols=['emi_eligibility','max_monthly_emi']):
    return [col for col in data.columns if col not in target_cols]

# ================== 4️⃣ Split & Scale ==================
def split_and_scale_data(data):
    X = data.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
    y_class = data['emi_eligibility']
    y_reg = data['max_monthly_emi']
    X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=0.3, random_state=42, stratify=y_class)
    X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp, test_size=0.5, random_state=42, stratify=y_class_temp)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train,X_val,X_test,y_class_train,y_class_val,y_class_test,y_reg_train,y_reg_val,y_reg_test,scaler

# ================== 5️⃣ EDA ==================
def exploratory_data_analysis(data):
    st.subheader("Dataset Overview")
    st.dataframe(data.head())
    st.subheader("Statistical Summary")
    st.dataframe(data.describe().T)
    st.subheader("EMI Eligibility Distribution")
    st.bar_chart(data['emi_eligibility'].value_counts())
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ================== 6️⃣ MLflow Logging ==================
def log_classification_mlflow(model_name, model, X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_name=f"{model_name}_Classification"):
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, average='weighted')
        mlflow.log_params(model.get_params())
        mlflow.log_metric("F1_weighted", f1)
        mlflow.sklearn.log_model(model, name=model_name, input_example=X_val[:2], signature=infer_signature(X_val,preds))
    return f1

def log_regression_mlflow(model_name, model, X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_name=f"{model_name}_Regression"):
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        r2 = r2_score(y_val, preds)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, name=model_name, input_example=X_val[:2], signature=infer_signature(X_val,preds))
    return r2

# ================== 7️⃣ Full Pipeline Wrapper ==================
def run_full_emi_pipeline(file_path):
    # Ensure 'models' directory exists
    os.makedirs("models", exist_ok=True)

    # Load & clean
    data = load_and_clean_data(file_path)
    data = feature_engineering(data)
    feature_names = detect_features(data)
    
    # ML Prep
    X_train,X_val,X_test,y_class_train,y_class_val,y_class_test,y_reg_train,y_reg_val,y_reg_test,scaler = split_and_scale_data(data)
    mlflow.set_experiment("EMI_Prediction_Experiment")
    
    # Classification
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoostClassifier": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    }
    best_clf, best_clf_score, best_clf_name = None, -1, ""
    for name, model in classifiers.items():
        score = log_classification_mlflow(name, model, X_train, X_val, y_class_train, y_class_val)
        if score > best_clf_score:
            best_clf_score = score
            best_clf = model
            best_clf_name = name
        # Save every classifier
        joblib.dump(model, f"models/{name}.pkl")
    # Regression
    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoostRegressor": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    }
    best_reg, best_reg_score, best_reg_name = None, -float('inf'), ""
    for name, model in regressors.items():
        score = log_regression_mlflow(name, model, X_train, X_val, y_reg_train, y_reg_val)
        if score > best_reg_score:
            best_reg_score = score
            best_reg = model
            best_reg_name = name
        # Save every regressor
        joblib.dump(model, f"models/{name}.pkl")
    
    # Save models
    # clf_model_path = f"best_classifier_{best_clf_name}.pkl"
    # reg_model_path = f"best_regressor_{best_reg_name}.pkl"
    # joblib.dump(best_clf, clf_model_path)
    # joblib.dump(best_reg, reg_model_path)

    # ================== Save Best Models Separately ==================
    joblib.dump(best_clf, f"models/best_classifier_{best_clf_name}.pkl")
    joblib.dump(best_reg, f"models/best_regressor_{best_reg_name}.pkl")
    
    # ================== Save Feature Names ==================
    joblib.dump(feature_names, "models/feature_names.pkl")

    # Save processed data for Streamlit (after cleaning & feature engineering)
    joblib.dump(data, "models/processed_data.pkl")
    
    print("Pipeline finished. All models saved in 'models/' folder.")
    
    # Launch Streamlit
    # run_emi_streamlit_app_interactive(clf_model_path, reg_model_path, feature_names)

def launch_emi_streamlit():
    # Load saved models and data
    clf_model_path = sorted([f for f in os.listdir() if f.startswith("best_classifier")])[0]
    reg_model_path = sorted([f for f in os.listdir() if f.startswith("best_regressor")])[0]
    feature_names = joblib.load("feature_names.pkl")
    
    # Start Streamlit app
    run_emi_streamlit_app_interactive(clf_model_path, reg_model_path, feature_names)


# ================== 8️⃣ Interactive Streamlit ==================
def run_emi_streamlit_app_interactive(classifier_model_path, regressor_model_path, feature_names):
    clf_model = joblib.load(classifier_model_path)
    reg_model = joblib.load(regressor_model_path)
    st.set_page_config(page_title="EMI Prediction Dashboard", layout="wide")
    st.title("EMI Prediction Platform")
    
    uploaded_file = st.file_uploader("Upload EMI dataset CSV", type="csv")
    if uploaded_file:
        # data = pd.read_csv(uploaded_file)
        data = load_and_clean_data(uploaded_file)  # ✅ full preprocessing
        data = feature_engineering(data)           # ✅ derived features
        exploratory_data_analysis(data)            # EDA
        # data = feature_engineering(data)
        # exploratory_data_analysis(data)
        st.subheader("Predictions")
        preds_class = clf_model.predict(data[feature_names])
        preds_reg = reg_model.predict(data[feature_names])
        data["EMI_Eligibility_Pred"] = preds_class
        data["Max_EMI_Pred"] = preds_reg
        st.dataframe(data[['EMI_Eligibility_Pred','Max_EMI_Pred']].head())
        st.success("Predictions completed successfully!")

# ================== 9️⃣ Main ==================
if __name__ == "__main__":
    file_path = r"C:\Users\haris\OneDrive\Desktop\Guvi\Projects\EMI_Prediction\dataset\emi_prediction_dataset.csv"  # change to your dataset path
    run_full_emi_pipeline(file_path)
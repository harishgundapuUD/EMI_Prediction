# emi_full_pipeline_no_streamlit_v2.py
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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
def load_and_clean_data(file_path, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    data = pd.read_csv(file_path)
    data = data.drop_duplicates()
    
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

    # Save cleaned dataset
    joblib.dump(data, os.path.join(save_dir, "cleaned_data.pkl"))
    return data

# ================== 2️⃣ Feature Engineering ==================
def feature_engineering(data, save_dir="models"):
    data['debt_to_income'] = data['current_emi_amount'] / data['monthly_salary']
    data['expense_to_income'] = (data['school_fees'] + data['college_fees'] + data['travel_expenses'] + 
                                 data['groceries_utilities'] + data['other_monthly_expenses']) / data['monthly_salary']
    data['affordability_ratio'] = (data['monthly_salary'] - data['current_emi_amount'] - 
                                   data['monthly_rent'] - data['other_monthly_expenses']) / data['requested_amount']
    
    # Save feature-engineered dataset
    joblib.dump(data, os.path.join(save_dir, "feature_engineered_data.pkl"))
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

# ================== 5️⃣ EDA (Save Plots) ==================
def exploratory_data_analysis(data, save_dir="models/eda_plots"):
    os.makedirs(save_dir, exist_ok=True)
    # Save dataset head & summary
    data.head().to_csv(os.path.join(save_dir, "dataset_head.csv"), index=False)
    data.describe().T.to_csv(os.path.join(save_dir, "stat_summary.csv"))
    
    # EMI Eligibility Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="emi_eligibility", data=data)
    plt.title("EMI Eligibility Distribution")
    plt.savefig(os.path.join(save_dir, "emi_eligibility_distribution.png"))
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"))
    plt.close()

# # ================== 6️⃣ MLflow Logging ==================
# def log_classification_mlflow(model_name, model, X_train, X_val, y_train, y_val):
#     with mlflow.start_run(run_name=f"{model_name}_Classification"):
#         model.fit(X_train, y_train)
#         preds = model.predict(X_val)
#         f1 = f1_score(y_val, preds, average='weighted')
#         mlflow.log_params(model.get_params())
#         mlflow.log_metric("F1_weighted", f1)
#         mlflow.sklearn.log_model(model, name=model_name, input_example=X_val[:2], signature=infer_signature(X_val,preds))
#     return f1

# def log_regression_mlflow(model_name, model, X_train, X_val, y_train, y_val):
#     with mlflow.start_run(run_name=f"{model_name}_Regression"):
#         model.fit(X_train, y_train)
#         preds = model.predict(X_val)
#         r2 = r2_score(y_val, preds)
#         mlflow.log_params(model.get_params())
#         mlflow.log_metric("R2", r2)
#         mlflow.sklearn.log_model(model, name=model_name, input_example=X_val[:2], signature=infer_signature(X_val,preds))
#     return r2

# ================== 6️⃣ MLflow Logging (Full Metrics + Print) ==================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

def log_classification_mlflow(model_name, model, X_train, X_val, y_train, y_val):
    """Logs classification model metrics to MLflow and prints metrics to console."""
    with mlflow.start_run(run_name=f"{model_name}_Classification"):
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # Compute metrics
        accuracy = accuracy_score(y_val, preds)
        precision = precision_score(y_val, preds, average='weighted')
        recall = recall_score(y_val, preds, average='weighted')
        f1 = f1_score(y_val, preds, average='weighted')

        # MLflow logging
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "Accuracy": accuracy,
            "Precision_weighted": precision,
            "Recall_weighted": recall,
            "F1_weighted": f1
        })
        mlflow.sklearn.log_model(model, name=model_name, input_example=X_val[:2], signature=infer_signature(X_val, preds))

        # Print metrics
        print(f"\n[Classification] {model_name} Metrics:")
        print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-weighted: {f1:.4f}")

    return f1  # Return F1 for selecting best model

def log_regression_mlflow(model_name, model, X_train, X_val, y_train, y_val):
    """Logs regression model metrics to MLflow and prints metrics to console."""
    with mlflow.start_run(run_name=f"{model_name}_Regression"):
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # Compute metrics
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        mape = np.mean(np.abs((y_val - preds) / y_val)) * 100

        # MLflow logging
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape
        })
        mlflow.sklearn.log_model(model, name=model_name, input_example=X_val[:2], signature=infer_signature(X_val, preds))

        # Print metrics
        print(f"\n[Regression] {model_name} Metrics:")
        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")

    return r2  # Return R2 for selecting best model


# ================== 7️⃣ Full Pipeline Wrapper ==================
def run_full_emi_pipeline(file_path, mlflow_path="models/mlruns"):
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # Load, clean, feature engineer
    data = load_and_clean_data(file_path, save_dir)
    data = feature_engineering(data, save_dir)
    feature_names = detect_features(data)
    joblib.dump(feature_names, os.path.join(save_dir, "feature_names.pkl"))

    # Save EDA
    exploratory_data_analysis(data)

    # Split & scale
    X_train,X_val,X_test,y_class_train,y_class_val,y_class_test,y_reg_train,y_reg_val,y_reg_test,scaler = split_and_scale_data(data)

    mlflow.set_tracking_uri(mlflow_path)
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
        joblib.dump(model, os.path.join(save_dir, f"{name}.pkl"))
        if score > best_clf_score:
            best_clf_score = score
            best_clf = model
            best_clf_name = name

    # Regression
    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoostRegressor": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    }
    best_reg, best_reg_score, best_reg_name = None, -float('inf'), ""
    for name, model in regressors.items():
        score = log_regression_mlflow(name, model, X_train, X_val, y_reg_train, y_reg_val)
        joblib.dump(model, os.path.join(save_dir, f"{name}.pkl"))
        if score > best_reg_score:
            best_reg_score = score
            best_reg = model
            best_reg_name = name

    # Save best models
    joblib.dump(best_clf, os.path.join(save_dir, f"best_classifier_{best_clf_name}.pkl"))
    joblib.dump(best_reg, os.path.join(save_dir, f"best_regressor_{best_reg_name}.pkl"))
    joblib.dump(data, os.path.join(save_dir, "processed_data.pkl"))

    print("Pipeline finished. All models, EDA, and datasets saved in 'models/' folder.")

# ================== 8️⃣ Main ==================
if __name__ == "__main__":
    file_path = r"C:\Users\haris\OneDrive\Desktop\Guvi\Projects\EMI_Prediction\dataset\emi_prediction_dataset.csv"
    run_full_emi_pipeline(file_path)






# how to use the saved models for prediction
# import joblib
# import numpy as np
# import pandas as pd
# import os

# # ================== 1️⃣ Load all models ==================
# model_dir = "models"  # folder where all models are saved

# # Load classifiers and regressors
# classifier_files = [f for f in os.listdir(model_dir) if f.startswith("best_classifier") or f.endswith(".pkl") and "Classifier" in f]
# regressor_files = [f for f in os.listdir(model_dir) if f.startswith("best_regressor") or f.endswith(".pkl") and "Regressor" in f]

# classifiers = {os.path.splitext(f)[0]: joblib.load(os.path.join(model_dir, f)) for f in classifier_files}
# regressors = {os.path.splitext(f)[0]: joblib.load(os.path.join(model_dir, f)) for f in regressor_files}

# # Load feature names
# feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))

# # ================== 2️⃣ User Input ==================
# user_input_dict = {
#     'age': 30,
#     'gender': 1,
#     'marital_status': 0,
#     'education': 2,
#     'monthly_salary': 50000,
#     'years_of_employment': 5,
#     'company_type': 3,
#     'house_type': 2,
#     'monthly_rent': 15000,
#     'family_size': 3,
#     'dependents': 1,
#     'school_fees': 5000,
#     'college_fees': 0,
#     'travel_expenses': 3000,
#     'groceries_utilities': 8000,
#     'other_monthly_expenses': 2000,
#     'existing_loans': 0,
#     'current_emi_amount': 7000,
#     'credit_score': 750,
#     'bank_balance': 20000,
#     'emi_scenario_E-commerce': 1,
#     'emi_scenario_Home Appliances': 0,
#     'emi_scenario_Vehicle': 0,
#     'emi_scenario_Personal Loan': 0,
#     'emi_scenario_Education': 0,
#     'emergency_fund': 10000
# }

# X_input = pd.DataFrame([user_input_dict])
# X_input = X_input[feature_names]  # match training feature order

# # ================== 3️⃣ Choose Models ==================
# print("Available Classifiers:", list(classifiers.keys()))
# print("Available Regressors:", list(regressors.keys()))

# chosen_clf_name = input("Enter classifier name: ")
# chosen_reg_name = input("Enter regressor name: ")

# clf_model = classifiers[chosen_clf_name]
# reg_model = regressors[chosen_reg_name]

# # ================== 4️⃣ Prediction ==================
# class_mapping = {0: "Not Eligible", 1: "High Risk", 2: "Eligible"}

# # Classification
# try:
#     pred_probs = clf_model.predict_proba(X_input)
#     pred_class_index = np.argmax(pred_probs, axis=1)[0]
#     confidence = pred_probs[0, pred_class_index]
# except AttributeError:
#     pred_class_index = clf_model.predict(X_input)[0]
#     confidence = None

# pred_class_label = class_mapping[pred_class_index]

# # Regression
# pred_max_emi = reg_model.predict(X_input)[0]

# # ================== 5️⃣ Output ==================
# print(f"\nSelected Classifier: {chosen_clf_name}")
# print(f"Selected Regressor: {chosen_reg_name}")
# print(f"Predicted EMI Eligibility: {pred_class_label}")
# if confidence is not None:
#     print(f"Confidence: {confidence:.2f}")
# print(f"Predicted Maximum EMI Amount: ₹{pred_max_emi:.2f}")

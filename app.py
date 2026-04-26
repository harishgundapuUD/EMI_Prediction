import streamlit as st
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
from src.dataset_processing import DataPreprocessor


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# load config data
config = {}
with open("utils/config.json") as f:
    config = json.load(f)

if config:
    streamlit_data = config.get("streamlit_data", {})

# Load results
with open("trained_models/classification_models/model_metrics.json") as f:
    classification_results = json.load(f)

with open("trained_models/regression_models/model_metrics.json") as f:
    regression_results = json.load(f)

# Find best model
for name, data in classification_results["models"]["classification"].items():
    if data["bestmodel"] == "yes":
        classification_best_model = name
        classification_run_id = data["run_id"]
        break

for name, data in regression_results["models"]["regression"].items():
    if data["bestmodel"] == "yes":
        regression_best_model = name
        regression_run_id = data["run_id"]
        break


# ----------------------------
# CACHE MODEL LOADING
# ----------------------------
@st.cache_resource
def load_models():
    # Classification
    mlruns_path_cls = os.path.abspath("trained_models/classification_models/mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_path_cls.replace(os.sep, '/')}")

    classification_model = mlflow.sklearn.load_model(
        f"runs:/{classification_run_id}/{classification_best_model}"
    )

    # Regression
    mlruns_path_reg = os.path.abspath("trained_models/regression_models/mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_path_reg.replace(os.sep, '/')}")

    regression_model = mlflow.sklearn.load_model(
        f"runs:/{regression_run_id}/{regression_best_model}"
    )

    return classification_model, regression_model

classification_model, regression_model = load_models()


# # Set tracking URI
# mlruns_path = os.path.abspath("trained_models/classification_models/mlruns")
# mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")

# # Load model
# classification_model = mlflow.sklearn.load_model(f"runs:/{classification_run_id}/{classification_best_model}")

# mlruns_path = os.path.abspath("trained_models/regression_models/mlruns")
# mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
# regression_model = mlflow.sklearn.load_model(f"runs:/{regression_run_id}/{regression_best_model}")

all_columns = streamlit_data.get("all_entries", [])
limited_columns = streamlit_data.get("limits", {})
specific_entries = streamlit_data.get("specific_entries", {})


st.title("EMI Predictor")

input_data = {}
num_cols = 4
# ----------------------------
# Split fields
# ----------------------------
numeric_fields = []
categorical_fields = []

for column in all_columns:
    if column in specific_entries:
        categorical_fields.append(column)
    else:
        numeric_fields.append(column)

# ----------------------------
# NUMERIC SECTION (TOP)
# ----------------------------
st.subheader("Numeric Inputs")

cols = st.columns(num_cols)

for i, column in enumerate(numeric_fields):
    col = cols[i % num_cols]

    with col:
        if column in limited_columns:
            min_val, max_val = limited_columns[column]

            input_data[column] = st.number_input(
                column.replace("_", " ").title(),
                min_value=int(min_val),
                max_value=int(max_val),
                value=int(min_val),
                step=1
            )
        else:
            input_data[column] = st.number_input(
                column.replace("_", " ").title(),
                min_value=0,
                value=0,
                step=1000
            )

# ----------------------------
# CATEGORICAL SECTION (BOTTOM)
# ----------------------------
st.subheader("Categorical Inputs")

cols = st.columns(num_cols)

for i, column in enumerate(categorical_fields):
    col = cols[i % num_cols]

    with col:
        input_data[column] = st.selectbox(
            column.replace("_", " ").title(),
            specific_entries[column]
        )
# print(f"The input data is : {input_data}")
input_df = pd.DataFrame([input_data])

# ----------------------------
# CACHE PREPROCESSING
# ----------------------------
@st.cache_data
def preprocess_input(input_df):
    preprocessor = DataPreprocessor(data=input_df, config=config)
    preprocessor.preprocess(testing=True)
    converted = preprocessor.create_financial_features(testing=True)
    return converted

# preprocessor = DataPreprocessor(data=input_df, config=config)
# preprocessor.preprocess(testing=True)
# converted_data = preprocessor.create_financial_features(testing=True)

with open("utils/train_columns.json") as f:
    train_columns = json.load(f).get("train_columns", [])

# formated_data = {}

# for col in train_columns:
#     if col in converted_data.columns:
#         formated_data[col] = converted_data[col].iloc[0]
#     else:
#         formated_data[col] = 0

# formated_data = pd.DataFrame([formated_data])
# formated_data.replace([np.inf, -np.inf], 0, inplace=True)
# formated_data.fillna(0, inplace=True)

# formated_data = formated_data.drop(columns=config.get("drop_columns", []), errors='ignore')

# ----------------------------
# BUTTON
# ----------------------------
if st.button("Calculate"):
    # ----------------------------
    # INPUT VALIDATION
    # ----------------------------
    if input_data.get("monthly_salary", 0) == 0:
        st.error("⚠️ Please enter a valid Monthly Salary (cannot be 0).")
        st.stop()

    if input_data.get("existing_loans") == "yes" and input_data.get("requested_amount", 0) == 0:
        st.error("⚠️ Please enter a valid Requested Loan Amount (cannot be 0 when existing loans = yes).")
        st.stop()
    
    if input_data.get("house_type") == "rented" and input_data.get("monthly_rent", 0) == 0:
        st.error("⚠️ Please enter a valid Monthly Rent (cannot be 0 when house type = rented).")
        st.stop()
    
    if input_data.get("requested_tenure", 0) == 0:
        st.error("⚠️ Please enter a valid Requested Tenure (cannot be 0).")
        st.stop()
    
    if input_data.get("requested_amount", 0) == 0:
        st.error("⚠️ Please enter a valid Requested Loan Amount (cannot be 0).")
        st.stop()

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess
    converted_data = preprocess_input(input_df)

    formated_data = {}

    for col in train_columns:
        if col in converted_data.columns:
            formated_data[col] = converted_data[col].iloc[0]
        else:
            formated_data[col] = 0

    formated_data = pd.DataFrame([formated_data])

    formated_data = formated_data.drop(columns=config.get("drop_columns", []), errors='ignore')
    formated_data.replace([np.inf, -np.inf], 0, inplace=True)
    formated_data.fillna(0, inplace=True)
    
    # ---- Classification Prediction ----
    classification_pred = classification_model.predict(formated_data)

    # If you have label encoding mapping
    if "label_encoding_order" in config:
        label_map = config["label_encoding_order"]["emi_eligibility"]
        classification_result = label_map[classification_pred[0]]
    else:
        classification_result = classification_pred[0]

    # ---- Regression Prediction ----
    regression_pred = regression_model.predict(formated_data)

    # ----------------------------
    # DISPLAY RESULTS
    # ----------------------------
    st.markdown("---")
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"EMI Eligibility: {classification_result}")

    with col2:
        st.info(f"Max Monthly EMI: ₹ {round(regression_pred[0], 2)}")
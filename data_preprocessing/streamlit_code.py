# # streamlit_emi_predictor.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# # ================== 1️⃣ Load Models ==================
# def load_models(model_dir="models"):
#     models = {}
#     for file in os.listdir(model_dir):
#         if file.endswith(".pkl"):
#             model_name = file.replace(".pkl", "")
#             models[model_name] = joblib.load(os.path.join(model_dir, file))
#     return models

# # ================== 2️⃣ Define Input UI ==================
# def user_input_features():
#     st.sidebar.header("Enter Customer Details")
    
#     # Numeric inputs
#     age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
#     monthly_salary = st.sidebar.number_input("Monthly Salary (INR)", min_value=5000, max_value=1000000, value=50000)
#     years_of_employment = st.sidebar.number_input("Years of Employment", min_value=0, max_value=50, value=5)
#     monthly_rent = st.sidebar.number_input("Monthly Rent (INR)", min_value=0, max_value=100000, value=10000)
#     family_size = st.sidebar.number_input("Family Size", min_value=1, max_value=20, value=3)
#     dependents = st.sidebar.number_input("Dependents", min_value=0, max_value=10, value=0)
#     school_fees = st.sidebar.number_input("School Fees (INR)", min_value=0, max_value=500000, value=10000)
#     college_fees = st.sidebar.number_input("College Fees (INR)", min_value=0, max_value=500000, value=0)
#     travel_expenses = st.sidebar.number_input("Travel Expenses (INR)", min_value=0, max_value=50000, value=2000)
#     groceries_utilities = st.sidebar.number_input("Groceries & Utilities (INR)", min_value=0, max_value=100000, value=8000)
#     other_monthly_expenses = st.sidebar.number_input("Other Expenses (INR)", min_value=0, max_value=50000, value=2000)
#     current_emi_amount = st.sidebar.number_input("Existing EMI (INR)", min_value=0, max_value=100000, value=5000)
#     bank_balance = st.sidebar.number_input("Bank Balance (INR)", min_value=0, max_value=1000000, value=20000)
#     emergency_fund = st.sidebar.number_input("Emergency Fund (INR)", min_value=0, max_value=1000000, value=10000)
#     requested_amount = st.sidebar.number_input("Requested EMI Amount (INR)", min_value=5000, max_value=5000000, value=50000)
    
#     # Dropdown inputs
#     gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
#     marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
#     education = st.sidebar.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
#     employment_type = st.sidebar.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
#     company_type = st.sidebar.selectbox("Company Type", ["Startup", "Small", "Mid-size", "Large Indian", "MNC"])
#     house_type = st.sidebar.selectbox("House Type", ["Rented", "Family", "Own"])
#     existing_loans = st.sidebar.selectbox("Existing Loans", ["No", "Yes"])
#     emi_scenario = st.sidebar.selectbox("EMI Scenario", ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"])
    
#     # Create DataFrame for model
#     data = pd.DataFrame({
#         "age":[age],
#         "gender":[0 if gender=="Male" else 1],
#         "marital_status":[0 if marital_status=="Single" else 1],
#         "education":[["High School","Graduate","Post Graduate","Professional"].index(education)],
#         "monthly_salary":[monthly_salary],
#         "years_of_employment":[years_of_employment],
#         "monthly_rent":[monthly_rent],
#         "family_size":[family_size],
#         "dependents":[dependents],
#         "school_fees":[school_fees],
#         "college_fees":[college_fees],
#         "travel_expenses":[travel_expenses],
#         "groceries_utilities":[groceries_utilities],
#         "other_monthly_expenses":[other_monthly_expenses],
#         "current_emi_amount":[current_emi_amount],
#         "bank_balance":[bank_balance],
#         "emergency_fund":[emergency_fund],
#         "requested_amount":[requested_amount],
#         "company_type":[["Startup", "Small", "Mid-size", "Large Indian", "MNC"].index(company_type)],
#         "house_type":[["Rented", "Family", "Own"].index(house_type)],
#         "existing_loans":[0 if existing_loans=="No" else 1],
#         "employment_type_"+employment_type:[1]
#     })
    
#     # One-hot encode emi_scenario
#     for scenario in ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"]:
#         data[f"emi_scenario_{scenario}"] = 1 if emi_scenario==scenario else 0
    
#     # Fill missing employment_type columns
#     for col in ["employment_type_Private","employment_type_Government","employment_type_Self-employed"]:
#         if col not in data.columns:
#             data[col] = 0
    
#     return data

# # ================== 3️⃣ Prediction ==================
# def predict_emi(models, input_df):
#     # Load classifiers and regressors separately
#     classifier_models = {k:v for k,v in models.items() if "Classifier" in k}
#     regressor_models = {k:v for k,v in models.items() if "Regressor" in k}
    
#     st.subheader("Select Model for Prediction")
#     clf_name = st.selectbox("Classifier Model", list(classifier_models.keys()))
#     reg_name = st.selectbox("Regressor Model", list(regressor_models.keys()))
    
#     # Classification Prediction
#     clf_model = classifier_models[clf_name]
#     preds_class_prob = clf_model.predict_proba(input_df)
#     pred_class_idx = np.argmax(preds_class_prob, axis=1)[0]
#     class_mapping = {0:"Not_Eligible", 1:"High_Risk", 2:"Eligible"}
#     pred_class_label = class_mapping[pred_class_idx]
    
#     # Regression Prediction
#     reg_model = regressor_models[reg_name]
#     pred_emi = reg_model.predict(input_df)[0]
    
#     # Show results
#     st.subheader("Predictions")
#     st.write("**EMI Eligibility:**", pred_class_label)
#     st.write("**Maximum EMI (INR):**", round(pred_emi,2))

# # ================== 4️⃣ Streamlit App ==================
# def run_streamlit_app():
#     st.title("EMI Prediction Dashboard")
    
#     # Load models
#     models = load_models("models")  # path to your saved pkl models
    
#     # Get user input
#     input_df = user_input_features()
    
#     # Predict
#     predict_emi(models, input_df)

# # ================== 5️⃣ Main ==================
# if __name__=="__main__":
#     run_streamlit_app()

























# # streamlit_emi_table.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# # ================== Load Models ==================
# def load_models(model_dir="models"):
#     models = {}
#     for file in os.listdir(model_dir):
#         if file.endswith(".pkl"):
#             model_name = file.replace(".pkl", "")
#             models[model_name] = joblib.load(os.path.join(model_dir, file))
#     return models

# # ================== Collect Inputs ==================
# def get_user_inputs():
#     st.sidebar.header("Customer Details")
    
#     inputs = {}
#     inputs["Age"] = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
#     inputs["Monthly Salary (INR)"] = st.sidebar.number_input("Monthly Salary", min_value=5000, max_value=1000000, value=50000)
#     inputs["Years of Employment"] = st.sidebar.number_input("Years of Employment", min_value=0, max_value=50, value=5)
#     inputs["Monthly Rent (INR)"] = st.sidebar.number_input("Monthly Rent", min_value=0, max_value=100000, value=10000)
#     inputs["Family Size"] = st.sidebar.number_input("Family Size", min_value=1, max_value=20, value=3)
#     inputs["Dependents"] = st.sidebar.number_input("Dependents", min_value=0, max_value=10, value=0)
#     inputs["School Fees (INR)"] = st.sidebar.number_input("School Fees", min_value=0, max_value=500000, value=10000)
#     inputs["College Fees (INR)"] = st.sidebar.number_input("College Fees", min_value=0, max_value=500000, value=0)
#     inputs["Travel Expenses (INR)"] = st.sidebar.number_input("Travel Expenses", min_value=0, max_value=50000, value=2000)
#     inputs["Groceries & Utilities (INR)"] = st.sidebar.number_input("Groceries & Utilities", min_value=0, max_value=100000, value=8000)
#     inputs["Other Expenses (INR)"] = st.sidebar.number_input("Other Expenses", min_value=0, max_value=50000, value=2000)
#     inputs["Current EMI Amount (INR)"] = st.sidebar.number_input("Current EMI", min_value=0, max_value=100000, value=5000)
#     inputs["Bank Balance (INR)"] = st.sidebar.number_input("Bank Balance", min_value=0, max_value=1000000, value=20000)
#     inputs["Emergency Fund (INR)"] = st.sidebar.number_input("Emergency Fund", min_value=0, max_value=1000000, value=10000)
#     inputs["Requested EMI Amount (INR)"] = st.sidebar.number_input("Requested EMI Amount", min_value=5000, max_value=5000000, value=50000)
    
#     # Dropdowns
#     inputs["Gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
#     inputs["Marital Status"] = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
#     inputs["Education"] = st.sidebar.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
#     inputs["Employment Type"] = st.sidebar.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
#     inputs["Company Type"] = st.sidebar.selectbox("Company Type", ["Startup", "Small", "Mid-size", "Large Indian", "MNC"])
#     inputs["House Type"] = st.sidebar.selectbox("House Type", ["Rented", "Family", "Own"])
#     inputs["Existing Loans"] = st.sidebar.selectbox("Existing Loans", ["No", "Yes"])
#     inputs["EMI Scenario"] = st.sidebar.selectbox("EMI Scenario", ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"])
    
#     return inputs

# # ================== Display Table ==================
# def display_input_table(inputs):
#     st.subheader("Customer Input Parameters")
#     input_df = pd.DataFrame(list(inputs.items()), columns=["Parameter", "Value"])
#     st.table(input_df)

# # ================== Prepare Model Input ==================
# def prepare_model_input(inputs):
#     df = pd.DataFrame({
#         "age": [inputs["Age"]],
#         "monthly_salary": [inputs["Monthly Salary (INR)"]],
#         "years_of_employment": [inputs["Years of Employment"]],
#         "monthly_rent": [inputs["Monthly Rent (INR)"]],
#         "family_size": [inputs["Family Size"]],
#         "dependents": [inputs["Dependents"]],
#         "school_fees": [inputs["School Fees (INR)"]],
#         "college_fees": [inputs["College Fees (INR)"]],
#         "travel_expenses": [inputs["Travel Expenses (INR)"]],
#         "groceries_utilities": [inputs["Groceries & Utilities (INR)"]],
#         "other_monthly_expenses": [inputs["Other Expenses (INR)"]],
#         "current_emi_amount": [inputs["Current EMI Amount (INR)"]],
#         "bank_balance": [inputs["Bank Balance (INR)"]],
#         "emergency_fund": [inputs["Emergency Fund (INR)"]],
#         "requested_amount": [inputs["Requested EMI Amount (INR)"]],
#         "gender": [0 if inputs["Gender"]=="Male" else 1],
#         "marital_status": [0 if inputs["Marital Status"]=="Single" else 1],
#         "education": [["High School", "Graduate", "Post Graduate", "Professional"].index(inputs["Education"])],
#         "company_type": [["Startup", "Small", "Mid-size", "Large Indian", "MNC"].index(inputs["Company Type"])],
#         "house_type": [["Rented", "Family", "Own"].index(inputs["House Type"])],
#         "existing_loans": [0 if inputs["Existing Loans"]=="No" else 1],
#     })
    
#     # One-hot encode employment_type
#     for emp in ["Private","Government","Self-employed"]:
#         df[f"employment_type_{emp}"] = 1 if inputs["Employment Type"]==emp else 0
    
#     # One-hot encode emi_scenario
#     for scenario in ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"]:
#         df[f"emi_scenario_{scenario}"] = 1 if inputs["EMI Scenario"]==scenario else 0
    
#     return df

# # ================== Prediction ==================
# def predict(models, model_input):
#     classifier_models = {k:v for k,v in models.items() if "Classifier" in k}
#     regressor_models = {k:v for k,v in models.items() if "Regressor" in k}
    
#     st.subheader("Select Models for Prediction")
#     clf_name = st.selectbox("Classifier Model", list(classifier_models.keys()))
#     reg_name = st.selectbox("Regressor Model", list(regressor_models.keys()))
    
#     # Classification
#     clf_model = classifier_models[clf_name]
#     pred_class_prob = clf_model.predict_proba(model_input)
#     pred_class_idx = np.argmax(pred_class_prob, axis=1)[0]
#     class_mapping = {0:"Not_Eligible",1:"High_Risk",2:"Eligible"}
#     pred_class_label = class_mapping[pred_class_idx]
    
#     # Regression
#     reg_model = regressor_models[reg_name]
#     pred_emi = reg_model.predict(model_input)[0]
    
#     # Show results
#     st.subheader("Predictions")
#     st.write("**EMI Eligibility:**", pred_class_label)
#     st.write("**Maximum EMI (INR):**", round(pred_emi,2))

# # ================== Main Streamlit App ==================
# def run_app():
#     st.title("EMI Prediction Dashboard")
    
#     inputs = get_user_inputs()
#     display_input_table(inputs)
    
#     models = load_models("models")  # Ensure your .pkl models are in this folder
#     model_input = prepare_model_input(inputs)
    
#     predict(models, model_input)

# if __name__=="__main__":
#     run_app()


















# import streamlit as st
# import pandas as pd

# st.title("EMI Prediction Input Table")

# # Define the parameters and their options (dropdowns for categorical, empty for numeric)
# parameters = {
#     "age": "",
#     "gender": ["male", "female"],
#     "marital_status": ["single", "married"],
#     "education": ["High School", "Graduate", "Post Graduate", "Professional"],
#     "years_of_employment": "",
#     "family_size": "",
#     "dependents": "",
#     "monthly_salary": "",
#     "employment_type": ["Private", "Government", "Self-employed"],
#     "company_type": ["Startup", "Small", "Mid-size", "Large Indian", "MNC"],
#     "house_type": ["Rented", "Family", "Own"],
#     "monthly_rent": "",
#     "existing_loans": ["yes", "no"],
#     "credit_score": "",
#     "bank_balance": "",
#     "emergency_fund": "",
#     "requested_amount": "",
#     "requested_tenure": "",
#     "emi_scenario": ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"]
# }

# # Create a table-like input using columns
# cols = st.columns([2, 3])  # column 1 = labels, column 2 = inputs
# user_input = {}
# for param, options in parameters.items():
#     with cols[0]:
#         st.markdown(f"**{param}**")
#     with cols[1]:
#         if options:
#             user_input[param] = st.selectbox(f"{param}", options, key=param)
#         else:
#             user_input[param] = st.text_input(f"{param}", key=param)

# st.write("User Inputs Table:")
# st.table(pd.DataFrame({
#     "Parameter": list(user_input.keys()),
#     "Value": list(user_input.values())
# }))


























# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # ===================== Load Saved Models =====================
# # Make sure these are paths to your trained models
# clf_model = joblib.load("models/best_classifier_XGBoostClassifier.pkl")
# reg_model = joblib.load("models/best_regressor_RandomForestRegressor.pkl")
# feature_names = joblib.load("models/feature_names.pkl")

# # ===================== Parameter Setup =====================
# parameters = {
#     "age": np.nan,
#     "gender": ["male", "female"],
#     "marital_status": ["single", "married"],
#     "education": ["High School", "Graduate", "Post Graduate", "Professional"],
#     "years_of_employment": np.nan,
#     "family_size": np.nan,
#     "dependents": np.nan,
#     "monthly_salary": np.nan,
#     "employment_type": ["Private", "Government", "Self-employed"],
#     "company_type": ["Startup", "Small", "Mid-size", "Large Indian", "MNC"],
#     "house_type": ["Rented", "Family", "Own"],
#     "monthly_rent": np.nan,
#     "existing_loans": ["yes", "no"],
#     "credit_score": np.nan,
#     "bank_balance": np.nan,
#     "emergency_fund": np.nan,
#     "requested_amount": np.nan,
#     "requested_tenure": np.nan,
#     "emi_scenario": ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"]
# }

# st.title("EMI Prediction Dashboard")

# # ===================== Create Editable Table =====================
# # Prepare a dataframe for the table
# table_df = pd.DataFrame({
#     "Parameter": list(parameters.keys()),
#     "Value": ["" if isinstance(v, list) else v for v in parameters.values()]
# })

# # Use data_editor for editable table
# edited_df = st.data_editor(table_df, num_rows="dynamic")

# # ===================== Buttons =====================
# col1, col2 = st.columns(2)
# predict_emi = col1.button("Predict EMI")
# clear_table = col2.button("Clear")

# # Clear table
# if clear_table:
#     edited_df["Value"] = ""
#     st.experimental_rerun()

# # Predict EMI
# if predict_emi:
#     # Convert inputs to proper format
#     input_dict = {}
#     for idx, row in edited_df.iterrows():
#         val = row["Value"]
#         param = row["Parameter"]
#         # Handle categorical mappings
#         if param in ["gender", "marital_status", "education", "employment_type", "company_type",
#                      "house_type", "existing_loans", "emi_scenario"]:
#             input_dict[param] = val
#         else:
#             try:
#                 input_dict[param] = float(val)
#             except:
#                 input_dict[param] = 0  # fallback for missing numeric values

#     # ===================== Preprocessing =====================
#     # Example: map gender, marital_status, education etc.
#     gender_map = {"male":0, "female":1}
#     marital_map = {"single":0, "married":1}
#     existing_loans_map = {"no":0, "yes":1}
#     edu_map = {"High School":0, "Graduate":1, "Post Graduate":2, "Professional":3}
#     house_map = {"Rented":0, "Family":1, "Own":2}

#     input_dict["gender"] = gender_map.get(input_dict["gender"], 0)
#     input_dict["marital_status"] = marital_map.get(input_dict["marital_status"], 0)
#     input_dict["existing_loans"] = existing_loans_map.get(input_dict["existing_loans"], 0)
#     input_dict["education"] = edu_map.get(input_dict["education"], 0)
#     input_dict["house_type"] = house_map.get(input_dict["house_type"], 0)

#     # Convert to DataFrame
#     input_df = pd.DataFrame([input_dict])
#     # Reorder / select feature_names
#     input_df = input_df.reindex(columns=feature_names, fill_value=0)

#     # ===================== Prediction =====================
#     pred_class = clf_model.predict(input_df)
#     pred_class_label = {0:"Not_Eligible", 1:"High_Risk", 2:"Eligible"}[pred_class[0]]
#     pred_emi = reg_model.predict(input_df)[0]

#     # ===================== Display Result =====================
#     st.subheader("Prediction Results")
#     st.write(f"**Loan Eligibility:** {pred_class_label}")
#     st.write(f"**Predicted Maximum EMI Amount:** ₹{pred_emi:.2f}")





























# import streamlit as st

# st.title("EMI Prediction Dashboard")

# # Define parameters and types/options
# params = [
#     ("Age", "number"),
#     ("Monthly Salary", "number"),
#     ("Gender", ["Male", "Female"]),
#     ("Marital Status", ["Single", "Married"]),
#     ("Education", ["High School", "Graduate", "Post Graduate", "Professional"]),
#     ("Employment Type", ["Private", "Government", "Self-employed"]),
#     ("Years of Employment", "number"),
#     ("Company Type", ["Startup", "Small", "Mid-size", "Large Indian", "MNC"]),
#     ("House Type", ["Rented", "Family", "Own"]),
#     ("Monthly Rent", "number"),
#     ("Existing Loans", ["Yes", "No"]),
#     ("Emergency Fund", "number"),
#     ("EMI Scenario", ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"]),
#     ("Requested Amount", "number"),
#     ("Requested Tenure (months)", "number")
# ]

# user_inputs = {}

# st.subheader("Input Parameters")
# for param, options in params:
#     cols = st.columns([1,2])
#     cols[0].write(param)
#     if isinstance(options, list):
#         user_inputs[param] = cols[1].selectbox(f"{param}", options)
#     else:
#         user_inputs[param] = cols[1].number_input(f"{param}", value=0)

# # Buttons
# col1, col2 = st.columns(2)
# predict = col1.button("Predict EMI")
# clear = col2.button("Clear")

# if predict:
#     st.write("Prediction logic goes here!")
#     st.write("User inputs:", user_inputs)

























# import streamlit as st

# st.title("EMI Prediction Dashboard")

# # Define parameters and their input types / options
# params = [
#     ("Age", "number"),
#     ("Monthly Salary", "number"),
#     ("Gender", ["Male", "Female"]),
#     ("Marital Status", ["Single", "Married"]),
#     ("Education", ["High School", "Graduate", "Post Graduate", "Professional"]),
#     ("Employment Type", ["Private", "Government", "Self-employed"]),
#     ("Years of Employment", "number"),
#     ("Company Type", ["Startup", "Small", "Mid-size", "Large Indian", "MNC"]),
#     ("House Type", ["Rented", "Family", "Own"]),
#     ("Monthly Rent", "number"),
#     ("Existing Loans", ["Yes", "No"]),
#     ("Emergency Fund", "number"),
#     ("EMI Scenario", ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"]),
#     ("Requested Amount", "number"),
#     ("Requested Tenure (months)", "number")
# ]

# st.subheader("Input Parameters")

# # Dictionary to store user input values
# user_inputs = {}

# # Loop to create table-like layout
# for param, options in params:
#     col1, col2 = st.columns([1, 2])
#     col1.markdown(f"**{param}**")  # Parameter name in left column
#     if isinstance(options, list):
#         user_inputs[param] = col2.selectbox(f"{param}", options, key=param)
#     else:
#         user_inputs[param] = col2.number_input(f"{param}", value=0, key=param)

# # Buttons
# col1, col2 = st.columns([1, 1])
# predict = col1.button("Predict EMI")
# clear = col2.button("Clear Inputs")

# if predict:
#     # Example: show the inputs as table
#     st.subheader("User Input Table")
#     st.table(user_inputs)

#     # Here, you would convert `user_inputs` to DataFrame, preprocess, and feed to models
#     # Example:
#     # df_input = preprocess_user_input(user_inputs)
#     # eligibility_pred = clf_model.predict(df_input)
#     # emi_pred = reg_model.predict(df_input)
#     # st.success(f"Eligibility: {eligibility_pred}, Predicted EMI: {emi_pred}")

























# import streamlit as st
# import pandas as pd

# st.title("EMI Prediction Dashboard")

# # Define parameters and options
# params = [
#     ("Age", "number"),
#     ("Monthly Salary", "number"),
#     ("Gender", ["Male", "Female"]),
#     ("Marital Status", ["Single", "Married"]),
#     ("Education", ["High School", "Graduate", "Post Graduate", "Professional"]),
#     ("Employment Type", ["Private", "Government", "Self-employed"]),
#     ("Years of Employment", "number"),
#     ("Company Type", ["Startup", "Small", "Mid-size", "Large Indian", "MNC"]),
#     ("House Type", ["Rented", "Family", "Own"]),
#     ("Monthly Rent", "number"),
#     ("Existing Loans", ["Yes", "No"]),
#     ("Emergency Fund", "number"),
#     ("EMI Scenario", ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"]),
#     ("Requested Amount", "number"),
#     ("Requested Tenure (months)", "number")
# ]

# # Build initial DataFrame
# df_input = pd.DataFrame({
#     "Parameter": [p[0] for p in params],
#     "Value": [None]*len(params)
# })

# # Replace table editing with st.data_editor
# edited_df = st.data_editor(df_input, num_rows="dynamic")

# # Buttons
# col1, col2 = st.columns(2)
# predict = col1.button("Predict EMI")
# clear = col2.button("Clear Inputs")

# if predict:
#     st.subheader("User Input Table")
#     st.dataframe(edited_df)

#     # Convert inputs to model-ready format here
#     # df_model_input = preprocess_user_input(edited_df)
#     # eligibility = clf_model.predict(df_model_input)
#     # emi = reg_model.predict(df_model_input)
#     # st.success(f"Eligibility: {eligibility}, Predicted EMI: {emi}")


























# import streamlit as st
# import joblib
# import numpy as np

# # Load saved models and features
# best_clf = joblib.load("models/best_classifier_XGBoostClassifier.pkl")
# best_reg = joblib.load("models/best_regressor_RandomForestRegressor.pkl")
# feature_names = joblib.load("models/feature_names.pkl")

# # Define categorical options
# gender_options = ["Male", "Female"]
# marital_options = ["Single", "Married"]
# education_options = ["High School", "Graduate", "Post Graduate", "Professional"]
# employment_options = ["Private", "Government", "Self-employed"]
# house_type_options = ["Rented", "Family", "Own"]
# existing_loans_options = ["No", "Yes"]
# emi_scenario_options = ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"]

# st.title("EMI Prediction")

# # Create two-column layout for each parameter
# def user_input():
#     input_data = {}
    
#     def add_row(param_name, widget_func, *args, **kwargs):
#         col1, col2 = st.columns([1,2])
#         with col1:
#             st.markdown(f"**{param_name}**")
#         with col2:
#             input_data[param_name] = widget_func(*args, **kwargs)
    
#     add_row("Age", st.number_input, min_value=18, max_value=100, value=30)
#     add_row("Gender", st.selectbox, options=gender_options)
#     add_row("Marital Status", st.selectbox, options=marital_options)
#     add_row("Education", st.selectbox, options=education_options)
#     add_row("Monthly Salary (INR)", st.number_input, min_value=0, value=50000)
#     add_row("Years of Employment", st.number_input, min_value=0, value=5)
#     add_row("Employment Type", st.selectbox, options=employment_options)
#     add_row("Company Type", st.text_input, value="Mid-size")
#     add_row("House Type", st.selectbox, options=house_type_options)
#     add_row("Monthly Rent", st.number_input, min_value=0, value=10000)
#     add_row("Family Size", st.number_input, min_value=1, value=3)
#     add_row("Dependents", st.number_input, min_value=0, value=1)
#     add_row("School Fees", st.number_input, min_value=0, value=0)
#     add_row("College Fees", st.number_input, min_value=0, value=0)
#     add_row("Travel Expenses", st.number_input, min_value=0, value=2000)
#     add_row("Groceries & Utilities", st.number_input, min_value=0, value=5000)
#     add_row("Other Monthly Expenses", st.number_input, min_value=0, value=1000)
#     add_row("Existing Loans", st.selectbox, options=existing_loans_options)
#     add_row("Current EMI Amount", st.number_input, min_value=0, value=0)
#     add_row("Credit Score", st.number_input, min_value=300, max_value=850, value=700)
#     add_row("Bank Balance", st.number_input, min_value=0, value=20000)
#     add_row("Emergency Fund", st.number_input, min_value=0, value=10000)
#     add_row("Requested Amount", st.number_input, min_value=0, value=50000)
#     add_row("Requested Tenure (months)", st.number_input, min_value=1, value=12)
#     add_row("EMI Scenario", st.selectbox, options=emi_scenario_options)

#     return input_data

# input_dict = user_input()

# col1, col2 = st.columns([1,1])
# with col1:
#     predict_btn = st.button("Predict EMI")
# with col2:
#     clear_btn = st.button("Clear Inputs")

# if clear_btn:
#     st.experimental_rerun()

# if predict_btn:
#     # Prepare data for model
#     # Convert categorical inputs to numeric encoding similar to training
#     mapping = {
#         "Gender": {"Male":0,"Female":1},
#         "Marital Status":{"Single":0,"Married":1},
#         "Education":{"High School":0,"Graduate":1,"Post Graduate":2,"Professional":3},
#         "Existing Loans":{"No":0,"Yes":1},
#         "EMI Scenario":{
#             "E-commerce Shopping EMI":0,
#             "Home Appliances EMI":1,
#             "Vehicle EMI":2,
#             "Personal Loan EMI":3,
#             "Education EMI":4
#         },
#         "House Type":{"Rented":0,"Family":1,"Own":2}
#     }

#     input_features = input_dict.copy()
#     for key in mapping:
#         input_features[key] = mapping[key][input_features[key]]
    
#     # Create dataframe with feature_names order
#     import pandas as pd
#     input_df = pd.DataFrame([[input_features.get(f, 0) for f in feature_names]], columns=feature_names)
    
#     # Predict classification and regression
#     pred_class_prob = best_clf.predict_proba(input_df)
#     pred_class_idx = np.argmax(pred_class_prob, axis=1)[0]
#     class_mapping = {0:"Not Eligible", 1:"High Risk", 2:"Eligible"}
#     pred_class_label = class_mapping[pred_class_idx]

#     pred_emi = best_reg.predict(input_df)[0]

#     st.success(f"Loan Eligibility: {pred_class_label}")
#     st.info(f"Predicted Maximum EMI: INR {pred_emi:.2f}")

















# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Load models and features
# best_clf = joblib.load("models/best_classifier_XGBoostClassifier.pkl")
# best_reg = joblib.load("models/best_regressor_RandomForestRegressor.pkl")
# feature_names = joblib.load("models/feature_names.pkl")

# # Categorical mappings
# gender_options = ["Male", "Female"]
# marital_options = ["Single", "Married"]
# education_options = ["High School", "Graduate", "Post Graduate", "Professional"]
# employment_options = ["Private", "Government", "Self-employed"]
# house_type_options = ["Rented", "Family", "Own"]
# existing_loans_options = ["No", "Yes"]
# emi_scenario_options = ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"]

# st.title("EMI Prediction")

# # Dictionary to store user inputs
# input_data = {}

# # Function to render a single row with parameter name and input
# # def render_row(param_name, widget_type="number", **kwargs):
# #     col1, col2 = st.columns([1,2])
# #     with col1:
# #         st.markdown(f"**{param_name}**")
# #     with col2:
# #         if widget_type == "number":
# #             input_data[param_name] = st.number_input(param_name, **kwargs)
# #         elif widget_type == "select":
# #             input_data[param_name] = st.selectbox(param_name, **kwargs)
# #         elif widget_type == "text":
# #             input_data[param_name] = st.text_input(param_name, **kwargs)

# def render_row(param_name, widget_type="number", **kwargs):
#     col1, col2 = st.columns([1,2])
#     with col1:
#         st.markdown(f"**{param_name}**")
#     with col2:
#         if widget_type == "number":
#             input_data[param_name] = st.number_input(
#                 "",  # empty label
#                 **kwargs,
#                 key=param_name  # unique key
#             )
#         elif widget_type == "select":
#             input_data[param_name] = st.selectbox(
#                 "",  # empty label
#                 **kwargs,
#                 key=param_name
#             )
#         elif widget_type == "text":
#             input_data[param_name] = st.text_input(
#                 "",  # empty label
#                 **kwargs,
#                 key=param_name
#             )



# # Render all input rows
# render_row("Age", widget_type="number", min_value=18, max_value=100, value=30)
# render_row("Gender", widget_type="select", options=gender_options)
# render_row("Marital Status", widget_type="select", options=marital_options)
# render_row("Education", widget_type="select", options=education_options)
# render_row("Monthly Salary (INR)", widget_type="number", min_value=0, value=50000)
# render_row("Years of Employment", widget_type="number", min_value=0, value=5)
# render_row("Employment Type", widget_type="select", options=employment_options)
# render_row("Company Type", widget_type="text", value="Mid-size")
# render_row("House Type", widget_type="select", options=house_type_options)
# render_row("Monthly Rent", widget_type="number", min_value=0, value=10000)
# render_row("Family Size", widget_type="number", min_value=1, value=3)
# render_row("Dependents", widget_type="number", min_value=0, value=1)
# render_row("School Fees", widget_type="number", min_value=0, value=0)
# render_row("College Fees", widget_type="number", min_value=0, value=0)
# render_row("Travel Expenses", widget_type="number", min_value=0, value=2000)
# render_row("Groceries & Utilities", widget_type="number", min_value=0, value=5000)
# render_row("Other Monthly Expenses", widget_type="number", min_value=0, value=1000)
# render_row("Existing Loans", widget_type="select", options=existing_loans_options)
# render_row("Current EMI Amount", widget_type="number", min_value=0, value=0)
# render_row("Credit Score", widget_type="number", min_value=300, max_value=850, value=700)
# render_row("Bank Balance", widget_type="number", min_value=0, value=20000)
# render_row("Emergency Fund", widget_type="number", min_value=0, value=10000)
# render_row("Requested Amount", widget_type="number", min_value=0, value=50000)
# render_row("Requested Tenure (months)", widget_type="number", min_value=1, value=12)
# render_row("EMI Scenario", widget_type="select", options=emi_scenario_options)

# col1, col2 = st.columns([1,1])
# with col1:
#     predict_btn = st.button("Predict EMI")
# with col2:
#     clear_btn = st.button("Clear Inputs")

# if clear_btn:
#     st.experimental_rerun()

# if predict_btn:
#     # Convert categorical inputs to numeric codes
#     mapping = {
#         "Gender": {"Male":0,"Female":1},
#         "Marital Status":{"Single":0,"Married":1},
#         "Education":{"High School":0,"Graduate":1,"Post Graduate":2,"Professional":3},
#         "Existing Loans":{"No":0,"Yes":1},
#         "EMI Scenario":{
#             "E-commerce Shopping EMI":0,
#             "Home Appliances EMI":1,
#             "Vehicle EMI":2,
#             "Personal Loan EMI":3,
#             "Education EMI":4
#         },
#         "House Type":{"Rented":0,"Family":1,"Own":2}
#     }

#     input_features = input_data.copy()
#     for key in mapping:
#         input_features[key] = mapping[key][input_features[key]]
    
#     # Convert to dataframe in correct order
#     input_df = pd.DataFrame([[input_features.get(f,0) for f in feature_names]], columns=feature_names)

#     # Predict
#     pred_class_prob = best_clf.predict_proba(input_df)
#     pred_class_idx = np.argmax(pred_class_prob, axis=1)[0]
#     class_mapping = {0:"Not Eligible", 1:"High Risk", 2:"Eligible"}
#     pred_class_label = class_mapping[pred_class_idx]

#     pred_emi = best_reg.predict(input_df)[0]

#     st.success(f"Loan Eligibility: {pred_class_label}")
#     st.info(f"Predicted Maximum EMI: INR {pred_emi:.2f}")


















# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Set page config to wide layout for better horizontal spacing
# st.set_page_config(layout="wide")

# # Load models and features
# best_clf = joblib.load("models/best_classifier_XGBoostClassifier.pkl")
# best_reg = joblib.load("models/best_regressor_RandomForestRegressor.pkl")
# feature_names = joblib.load("models/feature_names.pkl")

# # Categorical mappings
# gender_options = ["Male", "Female"]
# marital_options = ["Single", "Married"]
# education_options = ["High School", "Graduate", "Post Graduate", "Professional"]
# employment_options = ["Private", "Government", "Self-employed"]
# house_type_options = ["Rented", "Family", "Own"]
# existing_loans_options = ["No", "Yes"]
# emi_scenario_options = ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"]

# st.title("EMI Prediction")

# # Dictionary to store user inputs
# input_data = {}

# def render_row(param_name, widget_type="number", **kwargs):
#     col1, col2 = st.columns([1, 3])  # Adjusted widths for better spacing; increase if needed
#     with col1:
#         st.markdown(
#             f"<div style='display: flex; align-items: center; height: 100%;'><strong>{param_name}</strong></div>",
#             unsafe_allow_html=True
#         )
#     with col2:
#         if widget_type == "number":
#             input_data[param_name] = st.number_input(
#                 "",  # empty label
#                 **kwargs,
#                 key=param_name  # unique key
#             )
#         elif widget_type == "select":
#             input_data[param_name] = st.selectbox(
#                 "",  # empty label
#                 **kwargs,
#                 key=param_name
#             )
#         elif widget_type == "text":
#             input_data[param_name] = st.text_input(
#                 "",  # empty label
#                 **kwargs,
#                 key=param_name
#             )

# # Render all input rows
# render_row("Age", widget_type="number", min_value=18, max_value=100, value=30)
# render_row("Gender", widget_type="select", options=gender_options)
# render_row("Marital Status", widget_type="select", options=marital_options)
# render_row("Education", widget_type="select", options=education_options)
# render_row("Monthly Salary (INR)", widget_type="number", min_value=0, value=50000)
# render_row("Years of Employment", widget_type="number", min_value=0, value=5)
# render_row("Employment Type", widget_type="select", options=employment_options)
# render_row("Company Type", widget_type="text", value="Mid-size")
# render_row("House Type", widget_type="select", options=house_type_options)
# render_row("Monthly Rent", widget_type="number", min_value=0, value=10000)
# render_row("Family Size", widget_type="number", min_value=1, value=3)
# render_row("Dependents", widget_type="number", min_value=0, value=1)
# render_row("School Fees", widget_type="number", min_value=0, value=0)
# render_row("College Fees", widget_type="number", min_value=0, value=0)
# render_row("Travel Expenses", widget_type="number", min_value=0, value=2000)
# render_row("Groceries & Utilities", widget_type="number", min_value=0, value=5000)
# render_row("Other Monthly Expenses", widget_type="number", min_value=0, value=1000)
# render_row("Existing Loans", widget_type="select", options=existing_loans_options)
# render_row("Current EMI Amount", widget_type="number", min_value=0, value=0)
# render_row("Credit Score", widget_type="number", min_value=300, max_value=850, value=700)
# render_row("Bank Balance", widget_type="number", min_value=0, value=20000)
# render_row("Emergency Fund", widget_type="number", min_value=0, value=10000)
# render_row("Requested Amount", widget_type="number", min_value=0, value=50000)
# render_row("Requested Tenure (months)", widget_type="number", min_value=1, value=12)
# render_row("EMI Scenario", widget_type="select", options=emi_scenario_options)

# col1, col2 = st.columns([1,1])
# with col1:
#     predict_btn = st.button("Predict EMI")
# with col2:
#     clear_btn = st.button("Clear Inputs")

# if clear_btn:
#     st.experimental_rerun()

# if predict_btn:
#     # Convert categorical inputs to numeric codes
#     mapping = {
#         "Gender": {"Male":0,"Female":1},
#         "Marital Status":{"Single":0,"Married":1},
#         "Education":{"High School":0,"Graduate":1,"Post Graduate":2,"Professional":3},
#         "Existing Loans":{"No":0,"Yes":1},
#         "EMI Scenario":{
#             "E-commerce Shopping EMI":0,
#             "Home Appliances EMI":1,
#             "Vehicle EMI":2,
#             "Personal Loan EMI":3,
#             "Education EMI":4
#         },
#         "House Type":{"Rented":0,"Family":1,"Own":2}
#     }

#     input_features = input_data.copy()
#     for key in mapping:
#         input_features[key] = mapping[key][input_features[key]]
    
#     # Convert to dataframe in correct order
#     input_df = pd.DataFrame([[input_features.get(f,0) for f in feature_names]], columns=feature_names)

#     # Predict
#     pred_class_prob = best_clf.predict_proba(input_df)
#     pred_class_idx = np.argmax(pred_class_prob, axis=1)[0]
#     class_mapping = {0:"Not Eligible", 1:"High Risk", 2:"Eligible"}
#     pred_class_label = class_mapping[pred_class_idx]

#     pred_emi = best_reg.predict(input_df)[0]

#     st.success(f"Loan Eligibility: {pred_class_label}")
#     st.info(f"Predicted Maximum EMI: INR {pred_emi:.2f}")




















# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Set page config to wide layout for better horizontal spacing
# st.set_page_config(layout="wide")

# # Load models and features
# best_clf = joblib.load("models/best_classifier_XGBoostClassifier.pkl")
# best_reg = joblib.load("models/best_regressor_RandomForestRegressor.pkl")
# feature_names = joblib.load("models/feature_names.pkl")

# # Categorical mappings
# gender_options = ["Male", "Female"]
# marital_options = ["Single", "Married"]
# education_options = ["High School", "Graduate", "Post Graduate", "Professional"]
# employment_options = ["Private", "Government", "Self-employed"]
# house_type_options = ["Rented", "Family", "Own"]
# existing_loans_options = ["No", "Yes"]
# emi_scenario_options = ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"]

# st.title("EMI Prediction")

# # Dictionary to store user inputs
# input_data = {}

# def render_row(param_name, widget_type="number", **kwargs):
#     col1, col2 = st.columns([1, 4])  # Adjusted widths for better spacing
#     with col1:
#         st.markdown("##")  # Add vertical spacing to align label with input
#         st.markdown(f"**{param_name}**")
#     with col2:
#         params = {"key": param_name, "label_visibility": "collapsed"}
#         params.update(kwargs)
#         if widget_type == "number":
#             params.setdefault("label", " ")  # Dummy label
#             input_data[param_name] = st.number_input(**params)
#         elif widget_type == "select":
#             params.setdefault("label", " ")  # Dummy label
#             input_data[param_name] = st.selectbox(**params)
#         elif widget_type == "text":
#             params.setdefault("label", " ")  # Dummy label
#             input_data[param_name] = st.text_input(**params)

# # Render all input rows
# render_row("Age", widget_type="number", min_value=18, max_value=100, value=30)
# render_row("Gender", widget_type="select", options=gender_options)
# render_row("Marital Status", widget_type="select", options=marital_options)
# render_row("Education", widget_type="select", options=education_options)
# render_row("Monthly Salary (INR)", widget_type="number", min_value=0, value=50000)
# render_row("Years of Employment", widget_type="number", min_value=0, value=5)
# render_row("Employment Type", widget_type="select", options=employment_options)
# render_row("Company Type", widget_type="text", value="Mid-size")
# render_row("House Type", widget_type="select", options=house_type_options)
# render_row("Monthly Rent", widget_type="number", min_value=0, value=10000)
# render_row("Family Size", widget_type="number", min_value=1, value=3)
# render_row("Dependents", widget_type="number", min_value=0, value=1)
# render_row("School Fees", widget_type="number", min_value=0, value=0)
# render_row("College Fees", widget_type="number", min_value=0, value=0)
# render_row("Travel Expenses", widget_type="number", min_value=0, value=2000)
# render_row("Groceries & Utilities", widget_type="number", min_value=0, value=5000)
# render_row("Other Monthly Expenses", widget_type="number", min_value=0, value=1000)
# render_row("Existing Loans", widget_type="select", options=existing_loans_options)
# render_row("Current EMI Amount", widget_type="number", min_value=0, value=0)
# render_row("Credit Score", widget_type="number", min_value=300, max_value=850, value=700)
# render_row("Bank Balance", widget_type="number", min_value=0, value=20000)
# render_row("Emergency Fund", widget_type="number", min_value=0, value=10000)
# render_row("Requested Amount", widget_type="number", min_value=0, value=50000)
# render_row("Requested Tenure (months)", widget_type="number", min_value=1, value=12)
# render_row("EMI Scenario", widget_type="select", options=emi_scenario_options)

# col1, col2 = st.columns([1,1])
# with col1:
#     predict_btn = st.button("Predict EMI")
# with col2:
#     clear_btn = st.button("Clear Inputs")

# if clear_btn:
#     st.experimental_rerun()

# if predict_btn:
#     # Convert categorical inputs to numeric codes
#     mapping = {
#         "Gender": {"Male":0,"Female":1},
#         "Marital Status":{"Single":0,"Married":1},
#         "Education":{"High School":0,"Graduate":1,"Post Graduate":2,"Professional":3},
#         "Existing Loans":{"No":0,"Yes":1},
#         "EMI Scenario":{
#             "E-commerce Shopping EMI":0,
#             "Home Appliances EMI":1,
#             "Vehicle EMI":2,
#             "Personal Loan EMI":3,
#             "Education EMI":4
#         },
#         "House Type":{"Rented":0,"Family":1,"Own":2}
#     }

#     input_features = input_data.copy()
#     for key in mapping:
#         input_features[key] = mapping[key][input_features[key]]
    
#     # Convert to dataframe in correct order
#     input_df = pd.DataFrame([[input_features.get(f,0) for f in feature_names]], columns=feature_names)

#     # Predict
#     pred_class_prob = best_clf.predict_proba(input_df)
#     pred_class_idx = np.argmax(pred_class_prob, axis=1)[0]
#     class_mapping = {0:"Not Eligible", 1:"High Risk", 2:"Eligible"}
#     pred_class_label = class_mapping[pred_class_idx]

#     pred_emi = best_reg.predict(input_df)[0]

#     st.success(f"Loan Eligibility: {pred_class_label}")
#     st.info(f"Predicted Maximum EMI: INR {pred_emi:.2f}")














import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page config to wide layout for better horizontal spacing
# st.set_page_config(layout="wide")

# Load models and features
best_clf = joblib.load("models/best_classifier_XGBoostClassifier.pkl")
best_reg = joblib.load("models/best_regressor_RandomForestRegressor.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Categorical mappings
gender_options = ["Male", "Female"]
marital_options = ["Single", "Married"]
education_options = ["High School", "Graduate", "Post Graduate", "Professional"]
employment_options = ["Private", "Government", "Self-employed"]
house_type_options = ["Rented", "Family", "Own"]
existing_loans_options = ["No", "Yes"]
emi_scenario_options = ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"]

st.title("EMI Prediction")

# Dictionary to store user inputs
input_data = {}

def render_row(param_name, widget_type="number", **kwargs):
    col1, col2 = st.columns([1, 3])  # Adjusted widths for better spacing
    with col1:
        st.markdown(
            f"<div style='display: flex; align-items: center; height: 38px;'><strong>{param_name}</strong></div>",
            unsafe_allow_html=True
        )
    with col2:
        if widget_type == "number":
            input_data[param_name] = st.number_input(
                " ",  # space as label
                label_visibility="collapsed",
                key=param_name,
                **kwargs
            )
        elif widget_type == "select":
            input_data[param_name] = st.selectbox(
                " ",  # space as label
                label_visibility="collapsed",
                key=param_name,
                **kwargs
            )
        elif widget_type == "text":
            input_data[param_name] = st.text_input(
                " ",  # space as label
                label_visibility="collapsed",
                key=param_name,
                **kwargs
            )

# Render all input rows
render_row("Age", widget_type="number", min_value=18, max_value=100, value=30)
render_row("Gender", widget_type="select", options=gender_options)
render_row("Marital Status", widget_type="select", options=marital_options)
render_row("Education", widget_type="select", options=education_options)
render_row("Monthly Salary (INR)", widget_type="number", min_value=0, value=50000)
render_row("Years of Employment", widget_type="number", min_value=0, value=5)
render_row("Employment Type", widget_type="select", options=employment_options)
render_row("Company Type", widget_type="text", value="Mid-size")
render_row("House Type", widget_type="select", options=house_type_options)
render_row("Monthly Rent", widget_type="number", min_value=0, value=10000)
render_row("Family Size", widget_type="number", min_value=1, value=3)
render_row("Dependents", widget_type="number", min_value=0, value=1)
render_row("School Fees", widget_type="number", min_value=0, value=0)
render_row("College Fees", widget_type="number", min_value=0, value=0)
render_row("Travel Expenses", widget_type="number", min_value=0, value=2000)
render_row("Groceries & Utilities", widget_type="number", min_value=0, value=5000)
render_row("Other Monthly Expenses", widget_type="number", min_value=0, value=1000)
render_row("Existing Loans", widget_type="select", options=existing_loans_options)
render_row("Current EMI Amount", widget_type="number", min_value=0, value=0)
render_row("Credit Score", widget_type="number", min_value=300, max_value=850, value=700)
render_row("Bank Balance", widget_type="number", min_value=0, value=20000)
render_row("Emergency Fund", widget_type="number", min_value=0, value=10000)
render_row("Requested Amount", widget_type="number", min_value=0, value=50000)
render_row("Requested Tenure (months)", widget_type="number", min_value=1, value=12)
render_row("EMI Scenario", widget_type="select", options=emi_scenario_options)

# col1, col2 = st.columns([1,1])
# with col1:
#     predict_btn = st.button("Predict EMI")
# with col2:
#     clear_btn = st.button("Clear Inputs")


# Align buttons to the right by placing them under the input column
# col1, col2 = st.columns([1, 3])
# with col1:
#     pass  # Empty to align with labels
# with col2:
#     subcol1, subcol2 = st.columns(2)
#     with subcol1:
#         predict_btn = st.button("Predict EMI")
#     with subcol2:
#         clear_btn = st.button("Clear Inputs")




# Align buttons to the right under the input column
# col1, col2 = st.columns([1, 3])
# with col1:
#     pass  # Empty to align with labels
# with col2:
#     empty, subcol1, subcol2 = st.columns([3, 1, 1])
#     with subcol1:
#         predict_btn = st.button("Predict EMI", use_container_width=True)
#     with subcol2:
#         clear_btn = st.button("Clear Inputs", use_container_width=True)




# Align buttons to the right under the input column
col1, col2 = st.columns([1, 3])
with col1:
    pass  # Empty to align with labels
with col2:
    empty, subcol1, subcol2 = st.columns([1, 2, 2])  # Adjusted ratios to make buttons wider
    with subcol1:
        predict_btn = st.button("Predict EMI", use_container_width=True)
    with subcol2:
        clear_btn = st.button("Clear Inputs", use_container_width=True)



if clear_btn:
    st.experimental_rerun()

if predict_btn:
    # Convert categorical inputs to numeric codes
    mapping = {
        "Gender": {"Male":0,"Female":1},
        "Marital Status":{"Single":0,"Married":1},
        "Education":{"High School":0,"Graduate":1,"Post Graduate":2,"Professional":3},
        "Existing Loans":{"No":0,"Yes":1},
        "EMI Scenario":{
            "E-commerce Shopping EMI":0,
            "Home Appliances EMI":1,
            "Vehicle EMI":2,
            "Personal Loan EMI":3,
            "Education EMI":4
        },
        "House Type":{"Rented":0,"Family":1,"Own":2}
    }

    input_features = input_data.copy()
    for key in mapping:
        input_features[key] = mapping[key][input_features[key]]
    
    # Convert to dataframe in correct order
    input_df = pd.DataFrame([[input_features.get(f,0) for f in feature_names]], columns=feature_names)

    # Predict
    pred_class_prob = best_clf.predict_proba(input_df.values)[0]
    pred_class_idx = np.argmax(pred_class_prob, axis=1)[0]
    class_mapping = {0:"Not Eligible", 1:"High Risk", 2:"Eligible"}
    pred_class_label = class_mapping[pred_class_idx]

    pred_emi = best_reg.predict(input_df.values)[0]

    st.success(f"Loan Eligibility: {pred_class_label}")
    st.info(f"Predicted Maximum EMI: INR {pred_emi:.2f}")
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ================================================
# 1️⃣ Load All Models Dynamically
# ================================================
model_dir = "models"

classifier_files = [
    f for f in os.listdir(model_dir)
    if ("Classifier" in f and f.endswith(".pkl")) or ("logisticregression" in f.lower() and f.endswith(".pkl"))
]
regressor_files = [
    f for f in os.listdir(model_dir)
    if ("Regressor" in f and f.endswith(".pkl")) or ("linearregression" in f.lower() and f.endswith(".pkl"))
]

classifiers = {os.path.splitext(f)[0]: joblib.load(os.path.join(model_dir, f)) for f in classifier_files}
regressors = {os.path.splitext(f)[0]: joblib.load(os.path.join(model_dir, f)) for f in regressor_files}
feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))

# ================================================
# 2️⃣ Streamlit UI Setup
# ================================================
st.title("💰 EMI Prediction & Eligibility System")

st.markdown("This app predicts loan eligibility and estimated EMI using your financial profile.")

# ================================================
# 3️⃣ Input Widgets
# ================================================
gender_options = ["Male", "Female"]
marital_options = ["Single", "Married"]
education_options = ["High School", "Graduate", "Post Graduate", "Professional"]
employment_options = ["Private", "Government", "Self-employed"]
house_type_options = ["Rented", "Family", "Own"]
existing_loans_options = ["No", "Yes"]
emi_scenario_options = ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"]

input_data = {}

def render_row(param_name, widget_type="number", **kwargs):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(
            f"<div style='display: flex; align-items: center; height: 38px;'><strong>{param_name}</strong></div>",
            unsafe_allow_html=True
        )
    with col2:
        if widget_type == "number":
            input_data[param_name] = st.number_input(" ", label_visibility="collapsed", key=param_name, **kwargs)
        elif widget_type == "select":
            input_data[param_name] = st.selectbox(" ", label_visibility="collapsed", key=param_name, **kwargs)
        elif widget_type == "text":
            input_data[param_name] = st.text_input(" ", label_visibility="collapsed", key=param_name, **kwargs)

# Main user inputs
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

# Model selection
st.subheader("⚙️ Select Models for Prediction")
chosen_clf_name = st.selectbox("Choose Classifier", list(classifiers.keys()), index=0)
chosen_reg_name = st.selectbox("Choose Regressor", list(regressors.keys()), index=0)

# Buttons
col1, col2 = st.columns([1, 3])
with col2:
    subcol1, subcol2 = st.columns([2, 2])
    with subcol1:
        predict_btn = st.button("🔮 Predict EMI", use_container_width=True)
    with subcol2:
        clear_btn = st.button("🧹 Clear Inputs", use_container_width=True)

if clear_btn:
    st.rerun()

# ================================================
# 4️⃣ Preprocessing and Prediction
# ================================================
if predict_btn:
    # Encode categorical fields
    mapping = {
        "Gender": {"Male": 0, "Female": 1},
        "Marital Status": {"Single": 0, "Married": 1},
        "Education": {"High School": 0, "Graduate": 1, "Post Graduate": 2, "Professional": 3},
        "Existing Loans": {"No": 0, "Yes": 1},
        "House Type": {"Rented": 0, "Family": 1, "Own": 2},
    }

    # Initialize user input dictionary (numeric format)
    user_input_dict = {
        "age": input_data["Age"],
        "gender": mapping["Gender"][input_data["Gender"]],
        "marital_status": mapping["Marital Status"][input_data["Marital Status"]],
        "education": mapping["Education"][input_data["Education"]],
        "monthly_salary": input_data["Monthly Salary (INR)"],
        "years_of_employment": input_data["Years of Employment"],
        "company_type": hash(input_data["Company Type"]) % 5,  # Encode text into small numeric range
        "house_type": mapping["House Type"][input_data["House Type"]],
        "monthly_rent": input_data["Monthly Rent"],
        "family_size": input_data["Family Size"],
        "dependents": input_data["Dependents"],
        "school_fees": input_data["School Fees"],
        "college_fees": input_data["College Fees"],
        "travel_expenses": input_data["Travel Expenses"],
        "groceries_utilities": input_data["Groceries & Utilities"],
        "other_monthly_expenses": input_data["Other Monthly Expenses"],
        "existing_loans": mapping["Existing Loans"][input_data["Existing Loans"]],
        "current_emi_amount": input_data["Current EMI Amount"],
        "credit_score": input_data["Credit Score"],
        "bank_balance": input_data["Bank Balance"],
        "emergency_fund": input_data["Emergency Fund"],
        "requested_amount": input_data["Requested Amount"],
        "requested_tenure": input_data["Requested Tenure (months)"],
        "emi_scenario_E-commerce": 1 if input_data["EMI Scenario"] == "E-commerce Shopping EMI" else 0,
        "emi_scenario_Home Appliances": 1 if input_data["EMI Scenario"] == "Home Appliances EMI" else 0,
        "emi_scenario_Vehicle": 1 if input_data["EMI Scenario"] == "Vehicle EMI" else 0,
        "emi_scenario_Personal Loan": 1 if input_data["EMI Scenario"] == "Personal Loan EMI" else 0,
        "emi_scenario_Education": 1 if input_data["EMI Scenario"] == "Education EMI" else 0,
    }

    # Convert to dataframe
    X_input = pd.DataFrame([user_input_dict])
    X_input = X_input.reindex(columns=feature_names, fill_value=0)

    # Load selected models
    clf_model = classifiers[chosen_clf_name]
    reg_model = regressors[chosen_reg_name]

    class_mapping = {0: "Not Eligible", 1: "High Risk", 2: "Eligible"}

    # Classification
    try:
        pred_probs = clf_model.predict_proba(X_input.values)
        pred_class_index = np.argmax(pred_probs, axis=1)[0]
        confidence = pred_probs[0, pred_class_index]
    except AttributeError:
        pred_class_index = clf_model.predict(X_input.values)[0]
        confidence = None

    pred_class_label = class_mapping.get(pred_class_index, "Unknown")

    # Regression
    pred_max_emi = reg_model.predict(X_input.values)[0]

    # ================================================
    # 5️⃣ Display Results
    # ================================================
    st.success(f"🏦 **Loan Eligibility:** {pred_class_label}")
    if confidence is not None:
        st.write(f"**Confidence:** {confidence:.2f}")
    st.info(f"💸 **Predicted Maximum EMI:** ₹{pred_max_emi:,.2f}")
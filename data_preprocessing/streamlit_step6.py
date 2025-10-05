# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ================== 1️⃣ Load Models & Artifacts ==================
# MODEL_DIR = "models"
# classifiers = {os.path.splitext(f)[0]: joblib.load(os.path.join(MODEL_DIR, f))
#                for f in os.listdir(MODEL_DIR) if "Classifier" in f and f.endswith(".pkl")}
# regressors = {os.path.splitext(f)[0]: joblib.load(os.path.join(MODEL_DIR, f))
#               for f in os.listdir(MODEL_DIR) if "Regressor" in f and f.endswith(".pkl")}
# feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

# # Load cleaned and feature engineered dataset for EDA visualizations
# processed_data_path = os.path.join(MODEL_DIR, "processed_data.pkl")
# if os.path.exists(processed_data_path):
#     data = joblib.load(processed_data_path)
# else:
#     data = None

# # ================== 2️⃣ Streamlit Layout ==================
# st.set_page_config(page_title="💰 EMI Prediction System", layout="wide")
# st.title("💰 EMI Prediction & Maximum EMI Estimation")
# st.markdown("Predict your EMI eligibility and estimated maximum EMI based on your financial profile.")

# # Tabs: Prediction / EDA / About
# tabs = st.tabs(["🏦 Prediction", "📊 Exploratory Data Analysis", "ℹ️ About"])

# # ================== 3️⃣ Prediction Tab ==================
# with tabs[0]:
#     st.header("Enter Your Financial Details")

#     # Input widgets
#     input_data = {}
#     def render_row(param, widget_type="number", **kwargs):
#         col1, col2 = st.columns([1, 3])
#         with col1:
#             st.markdown(f"<strong>{param}</strong>", unsafe_allow_html=True)
#         with col2:
#             if widget_type=="number":
#                 input_data[param] = st.number_input("", key=param, **kwargs)
#             elif widget_type=="select":
#                 input_data[param] = st.selectbox("", key=param, **kwargs)
#             elif widget_type == "text":
#                 input_data[param] = st.text_input(" ", label_visibility="collapsed", key=param, **kwargs)

#     # Numeric & categorical inputs
#     render_row("Age", "number", min_value=18, max_value=100, value=30)
#     render_row("Gender", "select", options=["Male","Female"])
#     render_row("Marital Status", "select", options=["Single","Married"])
#     render_row("Education", "select", options=["High School","Graduate","Post Graduate","Professional"])
#     render_row("Monthly Salary (INR)", "number", min_value=0, value=50000)
#     render_row("Years of Employment", "number", min_value=0, value=5)
#     render_row("Employment Type", "select", options=["Private","Government","Self-employed"])
#     render_row("Company Type", "text")
#     render_row("House Type", "select", options=["Rented","Family","Own"])
#     render_row("Monthly Rent", "number", min_value=0, value=10000)
#     render_row("Family Size", "number", min_value=1, value=3)
#     render_row("Dependents", "number", min_value=0, value=1)
#     render_row("School Fees", "number", min_value=0, value=0)
#     render_row("College Fees", "number", min_value=0, value=0)
#     render_row("Travel Expenses", "number", min_value=0, value=2000)
#     render_row("Groceries & Utilities", "number", min_value=0, value=5000)
#     render_row("Other Monthly Expenses", "number", min_value=0, value=1000)
#     render_row("Existing Loans", "select", options=["No","Yes"])
#     render_row("Current EMI Amount", "number", min_value=0, value=0)
#     render_row("Credit Score", "number", min_value=300, max_value=900, value=700)
#     render_row("Bank Balance", "number", min_value=0, value=20000)
#     render_row("Emergency Fund", "number", min_value=0, value=10000)
#     render_row("Requested Amount", "number", min_value=0, value=50000)
#     render_row("Requested Tenure (months)", "number", min_value=1, value=12)
#     render_row("EMI Scenario", "select", options=["E-commerce","Home Appliances","Vehicle","Personal Loan","Education"])

#     st.subheader("⚙️ Choose Models")
#     chosen_clf_name = st.selectbox("Classifier", list(classifiers.keys()))
#     chosen_reg_name = st.selectbox("Regressor", list(regressors.keys()))

#     predict_btn, clear_btn = st.columns([1,1])
#     with predict_btn:
#         do_predict = st.button("🔮 Predict EMI")
#     with clear_btn:
#         do_clear = st.button("🧹 Clear Inputs")

#     if do_clear:
#         st.experimental_rerun()

#     if do_predict:
#         # ================== 4️⃣ Preprocessing ==================
#         mapping = {
#             "Gender": {"Male":0, "Female":1},
#             "Marital Status": {"Single":0, "Married":1},
#             "Education": {"High School":0,"Graduate":1,"Post Graduate":2,"Professional":3},
#             "Existing Loans": {"No":0,"Yes":1},
#             "House Type": {"Rented":0,"Family":1,"Own":2}
#         }

#         # Encode input dictionary
#         user_dict = {
#             "age": input_data["Age"],
#             "gender": mapping["Gender"][input_data["Gender"]],
#             "marital_status": mapping["Marital Status"][input_data["Marital Status"]],
#             "education": mapping["Education"][input_data["Education"]],
#             "monthly_salary": input_data["Monthly Salary (INR)"],
#             "years_of_employment": input_data["Years of Employment"],
#             "company_type": hash(input_data["Company Type"]) % 5,
#             "house_type": mapping["House Type"][input_data["House Type"]],
#             "monthly_rent": input_data["Monthly Rent"],
#             "family_size": input_data["Family Size"],
#             "dependents": input_data["Dependents"],
#             "school_fees": input_data["School Fees"],
#             "college_fees": input_data["College Fees"],
#             "travel_expenses": input_data["Travel Expenses"],
#             "groceries_utilities": input_data["Groceries & Utilities"],
#             "other_monthly_expenses": input_data["Other Monthly Expenses"],
#             "existing_loans": mapping["Existing Loans"][input_data["Existing Loans"]],
#             "current_emi_amount": input_data["Current EMI Amount"],
#             "credit_score": input_data["Credit Score"],
#             "bank_balance": input_data["Bank Balance"],
#             "emergency_fund": input_data["Emergency Fund"],
#             "requested_amount": input_data["Requested Amount"],
#             "requested_tenure": input_data["Requested Tenure (months)"],
#             "emi_scenario_E-commerce": 1 if input_data["EMI Scenario"]=="E-commerce" else 0,
#             "emi_scenario_Home Appliances": 1 if input_data["EMI Scenario"]=="Home Appliances" else 0,
#             "emi_scenario_Vehicle": 1 if input_data["EMI Scenario"]=="Vehicle" else 0,
#             "emi_scenario_Personal Loan": 1 if input_data["EMI Scenario"]=="Personal Loan" else 0,
#             "emi_scenario_Education": 1 if input_data["EMI Scenario"]=="Education" else 0,
#         }

#         X_input = pd.DataFrame([user_dict])
#         X_input = X_input.reindex(columns=feature_names, fill_value=0)

#         # ================== 5️⃣ Prediction ==================
#         clf_model = classifiers[chosen_clf_name]
#         reg_model = regressors[chosen_reg_name]
#         class_map = {0:"Not Eligible",1:"High Risk",2:"Eligible"}

#         try:
#             pred_probs = clf_model.predict_proba(X_input)
#             pred_index = np.argmax(pred_probs, axis=1)[0]
#             confidence = pred_probs[0,pred_index]
#         except AttributeError:
#             pred_index = clf_model.predict(X_input)[0]
#             confidence = None
#         pred_class = class_map.get(pred_index, "Unknown")
#         pred_emi = reg_model.predict(X_input)[0]

#         st.success(f"🏦 **Loan Eligibility:** {pred_class}")
#         if confidence:
#             st.write(f"**Confidence:** {confidence:.2f}")
#         st.info(f"💸 **Predicted Maximum EMI:** ₹{pred_emi:,.2f}")

# # ================== 6️⃣ EDA Tab ==================
# with tabs[1]:
#     st.header("Exploratory Data Analysis")
#     if data is not None:
#         st.subheader("EMI Eligibility Distribution")
#         fig, ax = plt.subplots()
#         sns.countplot(x="emi_eligibility", data=data, ax=ax)
#         st.pyplot(fig)

#         st.subheader("Correlation Heatmap")
#         fig, ax = plt.subplots(figsize=(10,8))
#         sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#         st.pyplot(fig)
#     else:
#         st.info("Processed dataset not found for EDA visualization.")

# # ================== 7️⃣ About Tab ==================
# with tabs[2]:
#     st.header("About")
#     st.markdown("""
#     This EMI Prediction app is built using:
#     - **Machine Learning Models:** Logistic Regression, Random Forest, XGBoost (classification & regression)
#     - **MLflow Tracking:** Model metrics, parameters, and artifacts
#     - **Streamlit UI:** Real-time prediction and EDA visualization
#     """)





















import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

# ================== 1️⃣ Load Models & Artifacts ==================
MODEL_DIR = "models"
classifiers = {os.path.splitext(f)[0]: joblib.load(os.path.join(MODEL_DIR, f))
               for f in os.listdir(MODEL_DIR) if "Classifier" in f and f.endswith(".pkl") or "logisticregression" in f.lower() and f.endswith(".pkl")}
regressors = {os.path.splitext(f)[0]: joblib.load(os.path.join(MODEL_DIR, f))
              for f in os.listdir(MODEL_DIR) if "Regressor" in f and f.endswith(".pkl") or "linearregression" in f.lower() and f.endswith(".pkl")}
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

# Load cleaned and feature engineered dataset for EDA visualizations
processed_data_path = os.path.join(MODEL_DIR, "processed_data.pkl")
if os.path.exists(processed_data_path):
    data = joblib.load(processed_data_path)
else:
    data = None


# ================== 2️⃣ MLflow Setup ==================
mlflow.set_tracking_uri(os.path.join(MODEL_DIR, "mlruns"))
experiment_name = "EMI_Prediction_Experiment"
experiment = mlflow.get_experiment_by_name(experiment_name)
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
clf_runs = runs_df[runs_df['tags.mlflow.runName'].str.contains("_Classification")]
reg_runs = runs_df[runs_df['tags.mlflow.runName'].str.contains("_Regression")]


# ================== 2️⃣ Streamlit Layout ==================
# st.set_page_config(page_title="💰 EMI Prediction System", layout="wide")
st.set_page_config(page_title="💰 EMI Prediction System")
st.title("💰 EMI Prediction & Maximum EMI Estimation")
st.markdown("Predict your EMI eligibility and estimated maximum EMI based on your financial profile.")

# Tabs: Prediction / EDA / About
tabs = st.tabs(["🏦 Prediction", "📊 EDA & Metrics", "ℹ️ About"])

# ================== 3️⃣ Prediction Tab ==================
with tabs[0]:
    st.header("Enter Your Financial Details")

    # Input widgets
    input_data = {}
    def render_row(param, widget_type="number", **kwargs):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"<strong>{param}</strong>", unsafe_allow_html=True)
        with col2:
            if widget_type=="number":
                input_data[param] = st.number_input(param, label_visibility="collapsed", key=param, **kwargs)
            elif widget_type=="select":
                input_data[param] = st.selectbox(param, label_visibility="collapsed", key=param, **kwargs)
            elif widget_type == "text":
                input_data[param] = st.text_input(param, label_visibility="collapsed", key=param, **kwargs)

    # Numeric & categorical inputs
    render_row("Age", "number", min_value=18, value=30)
    render_row("Gender", "select", options=["Male","Female"])
    render_row("Marital Status", "select", options=["Single","Married"])
    render_row("Education", "select", options=["High School","Graduate","Post Graduate","Professional"])
    render_row("Monthly Salary (INR)", "number", min_value=0, value=50000)
    render_row("Years of Employment", "number", min_value=0, value=5)
    render_row("Employment Type", "select", options=["Private","Government","Self-employed"])
    render_row("Company Type", "select", options=['Startup', 'Small', 'Mid-size', 'Large Indian', 'MNC'])
    render_row("House Type", "select", options=["Rented","Family","Own"])
    render_row("Monthly Rent", "number", min_value=0, value=10000)
    render_row("Family Size", "number", min_value=1, value=3)
    render_row("Dependents", "number", min_value=0, value=1)
    render_row("School Fees", "number", min_value=0, value=0)
    render_row("College Fees", "number", min_value=0, value=0)
    render_row("Travel Expenses", "number", min_value=0, value=2000)
    render_row("Groceries & Utilities", "number", min_value=0, value=5000)
    render_row("Other Monthly Expenses", "number", min_value=0, value=1000)
    render_row("Existing Loans", "select", options=["No","Yes"])
    render_row("Current EMI Amount", "number", min_value=0, value=0)
    render_row("Credit Score", "number", min_value=300, max_value=900, value=700)
    render_row("Bank Balance", "number", min_value=0, value=20000)
    render_row("Emergency Fund", "number", min_value=0, value=10000)
    render_row("Requested Amount", "number", min_value=0, value=50000)
    render_row("Requested Tenure (months)", "number", min_value=1, value=12)
    render_row("EMI Scenario", "select", options=["E-commerce","Home Appliances","Vehicle","Personal Loan","Education"])

    # st.subheader("⚙️ Choose Models")
    # chosen_clf_name = st.selectbox("Classifier", list(classifiers.keys()))
    # chosen_reg_name = st.selectbox("Regressor", list(regressors.keys()))

    # Model selection from MLflow runs
    st.subheader("⚙️ Choose Models")
    clf_options = list(classifiers.keys())
    reg_options = list(regressors.keys())
    chosen_clf_name = st.selectbox("Select Classifier Model", clf_options)
    chosen_reg_name = st.selectbox("Select Regressor Model", reg_options)

    # predict_btn, clear_btn = st.columns([1,1])
    # with predict_btn:
    #     do_predict = st.button("🔮 Predict EMI")
    # with clear_btn:
    #     do_clear = st.button("🧹 Clear Inputs")

    
    # Buttons
    col1, col2 = st.columns([1, 3])
    with col2:
        subcol1, subcol2 = st.columns([2, 2])
        with subcol1:
            do_predict = st.button("🔮 Predict EMI", use_container_width=True)
        with subcol2:
            do_clear = st.button("🧹 Clear Inputs", use_container_width=True)


    if do_clear:
        st.rerun()

    if do_predict:
        # ================== 4️⃣ Preprocessing ==================
        mapping = {
            "Gender": {"Male":0, "Female":1},
            "Marital Status": {"Single":0, "Married":1},
            "Education": {"High School":0,"Graduate":1,"Post Graduate":2,"Professional":3},
            "Existing Loans": {"No":0,"Yes":1},
            "House Type": {"Rented":0,"Family":1,"Own":2}
        }

        # Encode input dictionary
        user_dict = {
            "age": input_data["Age"],
            "gender": mapping["Gender"][input_data["Gender"]],
            "marital_status": mapping["Marital Status"][input_data["Marital Status"]],
            "education": mapping["Education"][input_data["Education"]],
            "monthly_salary": input_data["Monthly Salary (INR)"],
            "years_of_employment": input_data["Years of Employment"],
            "company_type": hash(input_data["Company Type"]) % 5,
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
            "emi_scenario_E-commerce": 1 if input_data["EMI Scenario"]=="E-commerce" else 0,
            "emi_scenario_Home Appliances": 1 if input_data["EMI Scenario"]=="Home Appliances" else 0,
            "emi_scenario_Vehicle": 1 if input_data["EMI Scenario"]=="Vehicle" else 0,
            "emi_scenario_Personal Loan": 1 if input_data["EMI Scenario"]=="Personal Loan" else 0,
            "emi_scenario_Education": 1 if input_data["EMI Scenario"]=="Education" else 0,
        }

        X_input = pd.DataFrame([user_dict])
        X_input = X_input.reindex(columns=feature_names, fill_value=0)

        # ================== 5️⃣ Prediction ==================
        # clf_model = classifiers[chosen_clf_name]
        # reg_model = regressors[chosen_reg_name]


        # Load models from MLflow artifacts
        # ================== 5️⃣ Prediction ==================
        # clf_run_id = classifiers[classifiers['tags.mlflow.runName']==chosen_clf_name].run_id.values[0]
        # reg_run_id = regressors[regressors['tags.mlflow.runName']==chosen_reg_name].run_id.values[0]

        # clf_uri = f"models/mlruns/{experiment.experiment_id}/{clf_run_id}/artifacts/{chosen_clf_name}"
        # reg_uri = f"models/mlruns/{experiment.experiment_id}/{reg_run_id}/artifacts/{chosen_reg_name}"
        # clf_model = mlflow.sklearn.load_model(clf_uri)
        # reg_model = mlflow.sklearn.load_model(reg_uri)

        clf_model = classifiers[chosen_clf_name]
        reg_model = regressors[chosen_reg_name]



        class_map = {0:"Not Eligible",1:"High Risk",2:"Eligible"}

        try:
            pred_probs = clf_model.predict_proba(X_input.values)
            pred_index = np.argmax(pred_probs, axis=1)[0]
            confidence = pred_probs[0,pred_index]
        except AttributeError:
            pred_index = clf_model.predict(X_input.values)[0]
            confidence = None
        pred_class = class_map.get(pred_index, "Unknown")
        pred_emi = reg_model.predict(X_input.values)[0]

        st.success(f"🏦 **Loan Eligibility:** {pred_class}")
        if confidence:
            st.write(f"**Confidence:** {confidence:.2f}")
        st.info(f"💸 **Predicted Maximum EMI:** ₹{pred_emi:,.2f}")

# ================== 6️⃣ EDA Tab ==================
with tabs[1]:
    st.header("Exploratory Data Analysis")
    if data is not None:
        st.subheader("EMI Eligibility Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="emi_eligibility", data=data, ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Processed dataset not found for EDA visualization.")

    st.subheader("📊 Classification Metrics (MLflow)")
    st.dataframe(clf_runs[['tags.mlflow.runName','metrics.F1_weighted','metrics.Accuracy',
                           'metrics.Precision_weighted','metrics.Recall_weighted']])
    st.subheader("📊 Regression Metrics (MLflow)")
    st.dataframe(reg_runs[['tags.mlflow.runName','metrics.R2','metrics.RMSE','metrics.MAE','metrics.MAPE']])

# ================== 7️⃣ About Tab ==================
with tabs[2]:
    st.header("About")
    # st.markdown("""
    # This EMI Prediction app is built using:
    # - **Machine Learning Models:** Logistic Regression, Random Forest, XGBoost (classification & regression)
    # - **MLflow Tracking:** Model metrics, parameters, and artifacts
    # - **Streamlit UI:** Real-time prediction and EDA visualization
    # """)

    st.markdown("""
    ### About
    This application predicts **loan eligibility** and **maximum EMI** based on your financial profile.
    - Developed with **Python, Streamlit, scikit-learn, XGBoost, and MLflow**
    - MLflow is used for **model tracking, metrics, and versioning**
    - Interactive **EDA** for insights on dataset distributions
    - Real-time predictions using the best performing models logged in MLflow
    """)



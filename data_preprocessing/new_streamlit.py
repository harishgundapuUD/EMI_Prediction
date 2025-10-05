import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

# ================== 1️⃣ Lazy Load Models & Artifacts ==================
MODEL_DIR = "models"

@st.cache_resource
def load_models():
    """Lazy load models to prevent memory issues"""
    classifiers = {}
    regressors = {}
    
    # Load classifiers
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            if "Classifier" in f or "logisticregression" in f.lower():
                try:
                    model_name = os.path.splitext(f)[0]
                    classifiers[model_name] = joblib.load(os.path.join(MODEL_DIR, f))
                    print(f"Loaded classifier: {model_name}")
                except Exception as e:
                    print(f"Error loading {f}: {e}")
    
    # Load regressors
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            if "Regressor" in f or "linearregression" in f.lower():
                try:
                    model_name = os.path.splitext(f)[0]
                    regressors[model_name] = joblib.load(os.path.join(MODEL_DIR, f))
                    print(f"Loaded regressor: {model_name}")
                except Exception as e:
                    print(f"Error loading {f}: {e}")
    
    return classifiers, regressors

@st.cache_resource
def load_feature_names():
    """Load feature names"""
    try:
        return joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    except:
        return None

@st.cache_resource
def load_processed_data():
    """Load processed data for EDA"""
    processed_data_path = os.path.join(MODEL_DIR, "processed_data.pkl")
    if os.path.exists(processed_data_path):
        return joblib.load(processed_data_path)
    return None

# Load data with error handling
try:
    classifiers, regressors = load_models()
    feature_names = load_feature_names()
    data = load_processed_data()
except Exception as e:
    st.error(f"Error loading models: {e}")
    classifiers, regressors = {}, {}
    feature_names = None
    data = None

# ================== 2️⃣ MLflow Setup ==================
@st.cache_resource
def load_mlflow_data():
    """Load MLflow data with error handling"""
    try:
        mlflow.set_tracking_uri(os.path.join(MODEL_DIR, "mlruns"))
        experiment_name = "EMI_Prediction_Experiment"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            clf_runs = runs_df[runs_df['tags.mlflow.runName'].str.contains("_Classification", na=False)]
            reg_runs = runs_df[runs_df['tags.mlflow.runName'].str.contains("_Regression", na=False)]
            return clf_runs, reg_runs
    except Exception as e:
        print(f"MLflow loading error: {e}")
    return pd.DataFrame(), pd.DataFrame()

clf_runs, reg_runs = load_mlflow_data()

# ================== 2️⃣ Streamlit Layout ==================
st.set_page_config(
    page_title="💰 EMI Prediction System", 
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("💰 EMI Prediction & Maximum EMI Estimation")
st.markdown("Predict your EMI eligibility and estimated maximum EMI based on your financial profile.")

# Initialize session state for inputs
if 'input_values' not in st.session_state:
    st.session_state.input_values = {
        "Age": 30,
        "Gender": "Male",
        "Marital Status": "Single",
        "Education": "Graduate",
        "Monthly Salary (INR)": 50000,
        "Years of Employment": 5,
        "Employment Type": "Private",
        "Company Type": "Mid-size",
        "House Type": "Rented",
        "Monthly Rent": 10000,
        "Family Size": 3,
        "Dependents": 1,
        "School Fees": 0,
        "College Fees": 0,
        "Travel Expenses": 2000,
        "Groceries & Utilities": 5000,
        "Other Monthly Expenses": 1000,
        "Existing Loans": "No",
        "Current EMI Amount": 0,
        "Credit Score": 700,
        "Bank Balance": 20000,
        "Emergency Fund": 10000,
        "Requested Amount": 50000,
        "Requested Tenure (months)": 12,
        "EMI Scenario": "Personal Loan"
    }

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Tabs: Prediction / EDA / About
tabs = st.tabs(["🏦 Prediction", "📊 EDA & Metrics", "ℹ️ About"])

# ================== 3️⃣ Prediction Tab ==================
with tabs[0]:
    st.header("Enter Your Financial Details")
    
    # Check if models are loaded
    if not classifiers or not regressors:
        st.error("❌ Models failed to load. Please check the model files and try again.")
        st.stop()
    
    # Input widgets function that updates session state
    def render_row(param, widget_type="number", **kwargs):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"<strong>{param}</strong>", unsafe_allow_html=True)
        with col2:
            if widget_type == "number":
                current_value = st.session_state.input_values[param]
                value = st.number_input(
                    param, 
                    label_visibility="collapsed", 
                    key=f"input_{param}",
                    value=current_value,
                    **kwargs
                )
                # Update session state immediately
                st.session_state.input_values[param] = value
                return value
            elif widget_type == "select":
                current_value = st.session_state.input_values[param]
                options = kwargs.get('options', [])
                value = st.selectbox(
                    param, 
                    label_visibility="collapsed", 
                    key=f"input_{param}",
                    options=options,
                    index=options.index(current_value) if current_value in options else 0
                )
                # Update session state immediately
                st.session_state.input_values[param] = value
                return value
    
    # Collect all inputs in two columns to reduce vertical space
    col1, col2 = st.columns(2)
    
    with col1:
        age = render_row("Age", "number", min_value=18)
        gender = render_row("Gender", "select", options=["Male","Female"])
        marital_status = render_row("Marital Status", "select", options=["Single","Married"])
        education = render_row("Education", "select", options=["High School","Graduate","Post Graduate","Professional"])
        monthly_salary = render_row("Monthly Salary (INR)", "number", min_value=0)
        years_of_employment = render_row("Years of Employment", "number", min_value=0)
        employment_type = render_row("Employment Type", "select", options=["Private","Government","Self-employed"])
        company_type = render_row("Company Type", "select", options=['Startup', 'Small', 'Mid-size', 'Large Indian', 'MNC'])
        house_type = render_row("House Type", "select", options=["Rented","Family","Own"])
        monthly_rent = render_row("Monthly Rent", "number", min_value=0)
        family_size = render_row("Family Size", "number", min_value=1)
        dependents = render_row("Dependents", "number", min_value=0)
    
    with col2:
        school_fees = render_row("School Fees", "number", min_value=0)
        college_fees = render_row("College Fees", "number", min_value=0)
        travel_expenses = render_row("Travel Expenses", "number", min_value=0)
        groceries_utilities = render_row("Groceries & Utilities", "number", min_value=0)
        other_monthly_expenses = render_row("Other Monthly Expenses", "number", min_value=0)
        existing_loans = render_row("Existing Loans", "select", options=["No","Yes"])
        current_emi_amount = render_row("Current EMI Amount", "number", min_value=0)
        credit_score = render_row("Credit Score", "number", min_value=300, max_value=900)
        bank_balance = render_row("Bank Balance", "number", min_value=0)
        emergency_fund = render_row("Emergency Fund", "number", min_value=0)
        requested_amount = render_row("Requested Amount", "number", min_value=0)
        requested_tenure = render_row("Requested Tenure (months)", "number", min_value=1)
        emi_scenario = render_row("EMI Scenario", "select", options=["E-commerce","Home Appliances","Vehicle","Personal Loan","Education"])

    # Model selection
    st.subheader("⚙️ Choose Models")
    clf_options = list(classifiers.keys())
    reg_options = list(regressors.keys())
    
    if 'chosen_clf' not in st.session_state:
        st.session_state.chosen_clf = clf_options[0] if clf_options else ""
    if 'chosen_reg' not in st.session_state:
        st.session_state.chosen_reg = reg_options[0] if reg_options else ""
    
    chosen_clf_name = st.selectbox(
        "Select Classifier Model", 
        clf_options, 
        key="clf_model",
        index=clf_options.index(st.session_state.chosen_clf) if st.session_state.chosen_clf in clf_options else 0
    )
    chosen_reg_name = st.selectbox(
        "Select Regressor Model", 
        reg_options, 
        key="reg_model",
        index=reg_options.index(st.session_state.chosen_reg) if st.session_state.chosen_reg in reg_options else 0
    )
    
    # Update session state with model selections
    st.session_state.chosen_clf = chosen_clf_name
    st.session_state.chosen_reg = chosen_reg_name
    
    # Buttons
    col1, col2 = st.columns([1, 3])
    with col2:
        subcol1, subcol2 = st.columns([2, 2])
        with subcol1:
            do_predict = st.button("🔮 Predict EMI", use_container_width=True, type="primary")
        with subcol2:
            do_clear = st.button("🧹 Clear Inputs", use_container_width=True)

    # Handle clear button
    if do_clear:
        st.session_state.input_values = {
            "Age": 30,
            "Gender": "Male",
            "Marital Status": "Single",
            "Education": "Graduate",
            "Monthly Salary (INR)": 50000,
            "Years of Employment": 5,
            "Employment Type": "Private",
            "Company Type": "Mid-size",
            "House Type": "Rented",
            "Monthly Rent": 10000,
            "Family Size": 3,
            "Dependents": 1,
            "School Fees": 0,
            "College Fees": 0,
            "Travel Expenses": 2000,
            "Groceries & Utilities": 5000,
            "Other Monthly Expenses": 1000,
            "Existing Loans": "No",
            "Current EMI Amount": 0,
            "Credit Score": 700,
            "Bank Balance": 20000,
            "Emergency Fund": 10000,
            "Requested Amount": 50000,
            "Requested Tenure (months)": 12,
            "EMI Scenario": "Personal Loan"
        }
        st.session_state.prediction_made = False
        st.session_state.prediction_result = None
        st.session_state.chosen_clf = clf_options[0] if clf_options else ""
        st.session_state.chosen_reg = reg_options[0] if reg_options else ""
        st.rerun()

    # Handle prediction
    if do_predict:
        with st.spinner("Processing your prediction..."):
            try:
                # ================== 4️⃣ Preprocessing ==================
                # More robust encoding mappings
                mapping = {
                    "Gender": {"Male": 0, "Female": 1},
                    "Marital Status": {"Single": 0, "Married": 1},
                    "Education": {"High School": 0, "Graduate": 1, "Post Graduate": 2, "Professional": 3},
                    "Existing Loans": {"No": 0, "Yes": 1},
                    "House Type": {"Rented": 0, "Family": 1, "Own": 2},
                    "Employment Type": {"Private": 0, "Government": 1, "Self-employed": 2},
                    "Company Type": {"Startup": 0, "Small": 1, "Mid-size": 2, "Large Indian": 3, "MNC": 4}
                }

                # Debug: Show current input values
                # st.write("### 🔍 Debug - Current Input Values:")
                # st.write(f"**House Type:** {st.session_state.input_values['House Type']} → Encoded as: {mapping['House Type'][st.session_state.input_values['House Type']]}")
                # st.write(f"**Age:** {st.session_state.input_values['Age']}")
                # st.write(f"**Monthly Salary:** {st.session_state.input_values['Monthly Salary (INR)']}")
                # st.write(f"**Credit Score:** {st.session_state.input_values['Credit Score']}")
                # st.write(f"**Requested Amount:** {st.session_state.input_values['Requested Amount']}")

                # Encode input dictionary using session state values
                user_dict = {
                    "age": st.session_state.input_values["Age"],
                    "gender": mapping["Gender"][st.session_state.input_values["Gender"]],
                    "marital_status": mapping["Marital Status"][st.session_state.input_values["Marital Status"]],
                    "education": mapping["Education"][st.session_state.input_values["Education"]],
                    "monthly_salary": st.session_state.input_values["Monthly Salary (INR)"],
                    "years_of_employment": st.session_state.input_values["Years of Employment"],
                    "employment_type": mapping["Employment Type"][st.session_state.input_values["Employment Type"]],
                    "company_type": mapping["Company Type"][st.session_state.input_values["Company Type"]],
                    "house_type": mapping["House Type"][st.session_state.input_values["House Type"]],  # Fixed encoding
                    "monthly_rent": st.session_state.input_values["Monthly Rent"],
                    "family_size": st.session_state.input_values["Family Size"],
                    "dependents": st.session_state.input_values["Dependents"],
                    "school_fees": st.session_state.input_values["School Fees"],
                    "college_fees": st.session_state.input_values["College Fees"],
                    "travel_expenses": st.session_state.input_values["Travel Expenses"],
                    "groceries_utilities": st.session_state.input_values["Groceries & Utilities"],
                    "other_monthly_expenses": st.session_state.input_values["Other Monthly Expenses"],
                    "existing_loans": mapping["Existing Loans"][st.session_state.input_values["Existing Loans"]],
                    "current_emi_amount": st.session_state.input_values["Current EMI Amount"],
                    "credit_score": st.session_state.input_values["Credit Score"],
                    "bank_balance": st.session_state.input_values["Bank Balance"],
                    "emergency_fund": st.session_state.input_values["Emergency Fund"],
                    "requested_amount": st.session_state.input_values["Requested Amount"],
                    "requested_tenure": st.session_state.input_values["Requested Tenure (months)"],
                    "emi_scenario_E-commerce": 1 if st.session_state.input_values["EMI Scenario"]=="E-commerce" else 0,
                    "emi_scenario_Home Appliances": 1 if st.session_state.input_values["EMI Scenario"]=="Home Appliances" else 0,
                    "emi_scenario_Vehicle": 1 if st.session_state.input_values["EMI Scenario"]=="Vehicle" else 0,
                    "emi_scenario_Personal Loan": 1 if st.session_state.input_values["EMI Scenario"]=="Personal Loan" else 0,
                    "emi_scenario_Education": 1 if st.session_state.input_values["EMI Scenario"]=="Education" else 0,
                }

                # Create input DataFrame
                X_input = pd.DataFrame([user_dict])
                
                # Debug: Show the input features before alignment
                # st.write("### 🔍 Input Features Before Alignment:")
                # st.write(X_input)
                
                # Align with feature names if available
                if feature_names is not None:
                    X_input = X_input.reindex(columns=feature_names, fill_value=0)
                    # st.write("### 🔍 Input Features After Alignment:")
                    # st.write(X_input[['house_type', 'age', 'monthly_salary', 'credit_score', 'requested_amount']])  # Show key features

                # ================== 5️⃣ Prediction ==================
                clf_model = classifiers[chosen_clf_name]
                reg_model = regressors[chosen_reg_name]

                class_map = {0: "Not Eligible", 1: "High Risk", 2: "Eligible"}

                # Make predictions
                clf_prediction = clf_model.predict(X_input.values)
                pred_index = clf_prediction[0]
                confidence = None
                
                # Try to get confidence scores if available
                try:
                    pred_probs = clf_model.predict_proba(X_input.values)
                    confidence = pred_probs[0, pred_index]
                    st.write(f"### 🔍 Prediction Probabilities: {pred_probs}")
                except (AttributeError, Exception) as e:
                    st.write(f"### 🔍 No probability scores available: {e}")
                
                pred_class = class_map.get(pred_index, "Unknown")
                pred_emi = reg_model.predict(X_input.values)[0]

                # Store prediction result in session state
                st.session_state.prediction_made = True
                st.session_state.prediction_result = {
                    "eligibility": pred_class,
                    "confidence": confidence,
                    "emi": pred_emi,
                    "inputs_used": st.session_state.input_values.copy(),
                    "house_type_encoded": mapping["House Type"][st.session_state.input_values["House Type"]]
                }

            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                import traceback
                st.write(f"Detailed error: {traceback.format_exc()}")
                st.session_state.prediction_made = False

    # Display prediction result if available
    if st.session_state.prediction_made and st.session_state.prediction_result:
        result = st.session_state.prediction_result
        st.success(f"🏦 **Loan Eligibility:** {result['eligibility']}")
        if result['confidence'] is not None:
            st.write(f"**Confidence:** {result['confidence']:.2f}")
        st.info(f"💸 **Predicted Maximum EMI:** ₹{result['emi']:,.2f}")
        
        # Show what inputs were used for this prediction
        # with st.expander("🔍 View detailed prediction information"):
            # st.write("**Inputs used for this prediction:**")
            # st.json(result['inputs_used'])
            # st.write(f"**House Type Encoded as:** {result['house_type_encoded']}")

# ================== 6️⃣ EDA Tab ==================
with tabs[1]:
    st.header("Exploratory Data Analysis")
    if data is not None:
        # Show house type distribution
        st.subheader("House Type Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="house_type", data=data, ax=ax)
        ax.set_xlabel("House Type (0=Rented, 1=Family, 2=Own)")
        st.pyplot(fig)

        st.subheader("EMI Eligibility Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="emi_eligibility", data=data, ax=ax)
        st.pyplot(fig)

        # Show relationship between house type and EMI eligibility
        st.subheader("EMI Eligibility by House Type")
        fig, ax = plt.subplots()
        sns.countplot(x="house_type", hue="emi_eligibility", data=data, ax=ax)
        ax.set_xlabel("House Type (0=Rented, 1=Family, 2=Own)")
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Processed dataset not found for EDA visualization.")

    st.subheader("📊 Classification Metrics (MLflow)")
    if not clf_runs.empty:
        st.dataframe(clf_runs[['tags.mlflow.runName','metrics.F1_weighted','metrics.Accuracy',
                               'metrics.Precision_weighted','metrics.Recall_weighted']])
    else:
        st.warning("No classification runs found in MLflow")
        
    st.subheader("📊 Regression Metrics (MLflow)")
    if not reg_runs.empty:
        st.dataframe(reg_runs[['tags.mlflow.runName','metrics.R2','metrics.RMSE','metrics.MAE','metrics.MAPE']])
    else:
        st.warning("No regression runs found in MLflow")

# ================== 7️⃣ About Tab ==================
with tabs[2]:
    st.header("About")
    st.markdown("""
    ### About
    This application predicts **loan eligibility** and **maximum EMI** based on your financial profile.
    - Developed with **Python, Streamlit, scikit-learn, XGBoost, and MLflow**
    - MLflow is used for **model tracking, metrics, and versioning**
    - Interactive **EDA** for insights on dataset distributions
    - Real-time predictions using the best performing models logged in MLflow
    
    ### 🏠 House Type Importance
    The model places significant importance on **House Type** as it indicates financial stability:
    - **Rented (0)**: Higher risk, lower eligibility
    - **Family (1)**: Moderate risk
    - **Own (2)**: Lower risk, higher eligibility
    """)
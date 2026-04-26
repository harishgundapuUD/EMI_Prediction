import streamlit as st
import json
import mlflow
import mlflow.sklearn
import pandas as pd

# Set MLflow tracking URI (same URI as during training)
mlflow.set_tracking_uri("file:///mlruns")  # Ensure it points to the correct location

# Load the results JSON file where the best model is marked
with open("models/model_metrics.json", "r") as f:
    results = json.load(f)

# Get the best regression model name from the results JSON (this was marked as "bestmodel": "yes")
best_model_name = None
for model_name, metrics in results["models"]["regression"].items():
    if metrics["bestmodel"] == "yes":
        best_model_name = model_name
        break

if best_model_name is None:
    st.error("No best regression model found. Please check training results.")
else:
    # Get MLflow run details
    run_id = results["run_details"]["regression"]["run_id"]
    run_path = results["run_details"]["regression"]["run_path"]

    # Load the best model using MLflow run_id and model name
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model/{best_model_name}")

    # Streamlit UI to get input from user
    st.title("Loan Eligibility Prediction")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    monthly_salary = st.number_input("Monthly Salary", min_value=0, max_value=1000000, value=50000)
    requested_amount = st.number_input("Requested Loan Amount", min_value=0, max_value=1000000, value=100000)
    
    # You can extend this form with more features (e.g., gender, marital status, etc.)
    
    # Collect input data into a DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'monthly_salary': [monthly_salary],
        'requested_amount': [requested_amount],
        # Add more features here as required
    })
    
    # Predict using the loaded model
    prediction = model.predict(input_data)
    
    # Show prediction result
    st.write(f"Predicted Loan Amount: {prediction[0]}")





########################################################
# classification code
import streamlit as st
import json
import mlflow
import mlflow.sklearn
import pandas as pd

# Set MLflow tracking URI (same URI as during training)
mlflow.set_tracking_uri("file:///mlruns")  # Ensure it points to the correct location

# Load the results JSON file where the best model is marked
with open("models/model_metrics.json", "r") as f:
    results = json.load(f)

# Get the best model name from the results JSON (this was marked as "bestmodel": "yes")
best_model_name = None
for model_name, metrics in results["models"]["classification"].items():
    if metrics["bestmodel"] == "yes":
        best_model_name = model_name
        break

if best_model_name is None:
    st.error("No best model found. Please check training results.")
else:
    # Get MLflow run details
    run_id = results["run_details"]["run_id"]
    run_path = results["run_details"]["run_path"]

    # Load the best model using MLflow run_id and model name
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model/{best_model_name}")

    # Streamlit UI to get input from user
    st.title("Loan Eligibility Prediction")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    monthly_salary = st.number_input("Monthly Salary", min_value=0, max_value=1000000, value=50000)
    requested_amount = st.number_input("Requested Loan Amount", min_value=0, max_value=1000000, value=100000)
    
    # You can extend this form with more features (e.g., gender, marital status, etc.)
    
    # Collect input data into a DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'monthly_salary': [monthly_salary],
        'requested_amount': [requested_amount],
        # Add more features here as required
    })
    
    # Predict using the loaded model
    prediction = model.predict(input_data)
    
    # Show prediction result
    if prediction[0] == 1:
        st.success("You are eligible for the loan!")
    else:
        st.error("You are not eligible for the loan.")

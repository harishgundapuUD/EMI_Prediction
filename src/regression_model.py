import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# CONFIG
# ----------------------------
# TARGET_COL = "target"  # change this to your label column
# MODEL_DIR = "models"  # This directory is used for saving the JSON file and models

config = {}
with open("utils/config.json", "r") as f:
    config = json.load(f)

if config:
        TARGET_COL = config["target_columns"]["regression"]
        MODEL_DIR = os.path.join(
            config["ml_model_dirs"]["base_dir"],
            config["ml_model_dirs"]["regression"]
        )
        model_metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
        drop_cols = config.get("drop_columns", [])

os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("dataset/processed_data.csv")  # replace with your CSV file path

X = df.drop(columns=drop_cols)
y = df[TARGET_COL]

# print(f"The x columns are : {X.columns}")
# print(f"The y column is : {TARGET_COL}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

# ----------------------------
# MODELS
# ----------------------------
models = {
    "linearregression": LinearRegression(),
    "decisiontreeregressor": DecisionTreeRegressor(max_depth=5, random_state=42),
    "randomforestregressor": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
}

# ----------------------------
# EVALUATION STORAGE
# ----------------------------

results = {
                "models": {"regression": {}}
            }

if os.path.exists(os.path.join(MODEL_DIR, "model_metrics.json")):
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "r") as f:
        existing_results = json.load(f)
    results = results | existing_results  # Merge with existing results

best_model_name = None
best_score = -1

# ----------------------------
# MLflow Setup
# ----------------------------
mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
mlflow.set_experiment("regression")

# Start the MLflow run for regression models
for name, model in models.items():
    print("===========================================================================")
    print(f"Training {name}...")
    with mlflow.start_run() as run:
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Multiply by 100 to get percentage

        # Log metrics with MLflow
        mlflow.log_metric(f"{name}_mse", mse)
        mlflow.log_metric(f"{name}_rmse", rmse)
        mlflow.log_metric(f"{name}_mae", mae)
        mlflow.log_metric(f"{name}_r2", r2)
        mlflow.log_metric(f"{name}_mape", mape)
        # Log model with MLflow
        mlflow.sklearn.log_model(model, name)

        # Store metrics in results dictionary for later comparison
        results["models"]["regression"][name] = {
            "run_id": run.info.run_id,
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "mape": float(mape),
            "bestmodel": "no"
        }

        # normalizing the metrics for custom scoring
        rmse_n = 1 / (1 + rmse)
        mae_n = 1 / (1 + mae)
        mape_n = 1 / (1 + mape)
        # r2_n = (r2 + 1) / 2   # maps [-1,1] → [0,1]
        r2_n = max(0, min(1, (r2 + 1) / 2))


        # Compute custom score (for regression, using R-squared as the score metric)
        # score = r2  # For regression, you can choose R2 as the scoring metric
        score = (
                    0.4 * r2_n +          # Emphasize normalized R-squared
                    0.3 * rmse_n +     # Emphasize normalized RMSE
                    0.2 * mae_n +      # Emphasize normalized MAE
                    0.1 * mape_n       # Emphasize normalized MAPE
                )

        # Select best regression model based on custom score
        if score > best_score:
            best_score = score
            best_model_name = name
        
    print(f"Trained {name} with custom score: {score:.4f}")
    print("===========================================================================")
    print("\n")

# Mark best regression model in the results
results["models"]["regression"][best_model_name]["bestmodel"] = "yes"

# ----------------------------
# SAVE MODELS AND METRICS
# ----------------------------
# Save models to disk
# for name, model in models.items():
#     joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

# ----------------------------
# SAVE MODELS AND METRICS
# ----------------------------
# Save the results to a JSON file with the MLflow run details
with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

# ----------------------------
# FINAL OUTPUT
# ----------------------------
print("Training complete.")
print("Best regression model:", best_model_name)
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient

# ----------------------------
# CONFIG
# ----------------------------
# TARGET_COL = "target"  # change this to your label column
# MODEL_DIR = "models"

config = {}
with open("utils/config.json", "r") as f:
    config = json.load(f)

if config:
        TARGET_COL = config["target_columns"]["classification"]
        MODEL_DIR = os.path.join(
            config["ml_model_dirs"]["base_dir"],
            config["ml_model_dirs"]["classification"]
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# MODELS
# ----------------------------
models = {
    "logisticregression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),
    "decisiontree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "randomforest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
}

# ----------------------------
# EVALUATION STORAGE
# ----------------------------
results = {
                "models": {"classification": {}}
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
mlflow.set_experiment("classification")

for name, model in models.items():
    print("===========================================================================")
    print(f"Training {name}...")
    with mlflow.start_run(run_name=name) as run:
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

            if y_prob.shape[1] == 2:
                # Binary classification → use probability of class 1
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                # Multiclass
                roc_auc = roc_auc_score(
                    y_test,
                    y_prob,
                    multi_class="ovr",
                    average="weighted"
                )
        else:
            roc_auc = None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

        # Log metrics with MLflow
        mlflow.log_metric(f"{name}_accuracy", accuracy)
        mlflow.log_metric(f"{name}_precision", precision)
        mlflow.log_metric(f"{name}_recall", recall)
        mlflow.log_metric(f"{name}_f1_score", f1)
        mlflow.log_metric(f"{name}_roc_auc", roc_auc)

        # Save the model with MLflow
        mlflow.sklearn.log_model(model, name)

        # Store metrics in results dictionary for later comparison
        results["models"]["classification"][name] = {
            "run_id": run.info.run_id,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "roc-auc": float(roc_auc),
            "bestmodel": "no"
        }

        # Compute custom score
        score = (
            0.4 * precision +
            0.3 * roc_auc +
            0.2 * f1 +
            0.1 * accuracy
        )

        # Select best model
        if score > best_score:
            best_score = score
            best_model_name = name
    print(f"Trained {name} with custom score: {score:.4f}")
    print("===========================================================================")
    print("\n")
    
# Mark best model in the results
results["models"]["classification"][best_model_name]["bestmodel"] = "yes"

# End the MLflow run (this is required to properly log all metrics and artifacts)
mlflow.end_run()

# ----------------------------
# SAVE MODELS AND METRICS
# ----------------------------
# Save models to disk
# for name, model in models.items():
#     joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

# Save the results to a JSON file
with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

# ----------------------------
# FINAL OUTPUT
# ----------------------------
print("Training complete.")
print("Best model:", best_model_name)



'''
Load the model:

import mlflow
model = mlflow.sklearn.load_model("models/randomforest")
Make predictions with the model based on user inputs (after preprocessing the input).
'''
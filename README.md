# EMI Prediction

A lightweight project to predict Equated Monthly Installments (EMIs) for loans using structured borrower and loan features. Includes data preprocessing, model training, evaluation, and inference scripts.

## Features

- Data cleaning and feature engineering
- Trainable regression model (configurable)
- Model evaluation with RMSE and MAE
- Reproducible training pipeline and saved model artifacts

## Dataset

- Expected CSV with columns such as: principal, annual_interest_rate, tenure_months, credit_score, income, and other relevant features.
- Include a train/test split before training.

## Requirements

- Check the requirements.txt file

## Installation

1. Clone the repository.
2. Create a virtual environment:

   python -m venv .venv

   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
3. Install dependencies:

   pip install -r requirements.txt

## Usage

* Use the src_code.py code to clean, pre-process and train the model.
* streamlit.py code is used to create the interactive app to predict the emi eligibility and emi amount.

## Model & Metrics

- Classification
  - Models - Logistic Regression, Random Forest, XG Boost
  - Metrics - Accuracy, Precision, Recall, F1 Score
- Regression
  - Models - Linear Regression, Random Forest, XG Boost
  - Metrics - RMSE (Root Mean Square Error), MAE (Mean Absolute Error), R2, MAPE (Mean Absolute Percentage Error)

## Project Structure

- dataset/                # raw dataset
- models/              # serialized model artifacts, EDA output and ml flow
- experiments/           # EDA and experiments
- requirements.txt
- README.md

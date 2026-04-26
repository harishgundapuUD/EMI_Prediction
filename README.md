# 💰 EMI Prediction System

A complete end-to-end Machine Learning project to predict:

* ✅ EMI Eligibility (Classification)
* ✅ Maximum Monthly EMI (Regression)

This project uses **MLflow for experiment tracking** and  **Streamlit for deployment** .

---

## 🚀 Features

* 🔹 Dual Model System:
  * Classification → EMI eligibility
  * Regression → Max monthly EMI
* 🔹 MLflow integration for tracking models
* 🔹 Feature engineering (financial ratios, affordability, etc.)
* 🔹 EDA analysis with visualizations
* 🔹 Dynamic Streamlit UI with validations
* 🔹 Automatic handling of missing features (one-hot encoding alignment)

---

## 🏗️ Project Structure

```bash
EMI_Prediction/

├── dataset/
│   ├── csv files
│
├── eda_analysis/
│   ├── analysis plots
│
├── src/
│   ├── dataset_processing.py
│   ├── classification_model.py
│   ├── regression_model.py
│   ├── eda_analysis.py
│
├── trained_models/
│   ├── classification_models/
│   │   ├── mlruns/
│   │   └── model_metrics.json
│   │
│   ├── regression_models/
│   │   ├── mlruns/
│   │   └── model_metrics.json
│
├── utils/
│   ├── config.json
│   └── train_columns.json
│
├── dataset/
│   └── processed_data.csv
│
├── app.py
├── README.md
```

---

## 🧠 ML Pipeline

### 1. Data Processing

* Cleaning & transformation
* Handling missing values
* Encoding categorical variables
* Feature engineering:
  * Expense-to-income ratio
  * Debt-to-income ratio
  * Savings rate
  * Financial buffer
  * EMI burden

---

### 2. Exploratory Data Analysis (EDA)

* Distribution plots
* Correlation analysis
* Feature relationships
* Insights stored in `eda_analysis/`

---

### 3. Model Training

* Models used:
  * Logistic Regression
  * Decision Tree
  * Random Forest
* MLflow used for:
  * Tracking experiments
  * Logging metrics
  * Saving models

---

### 4. Model Selection

Custom scoring formula:

```python
classification_score = (0.4 * precision) + (0.3 * roc_auc) + (0.2 * f1) + (0.1 * accuracy)
regression_score = (0.4 * r2) + (0.3 * rmse) + (0.2 * mae) + (0.1 * mape)
```

* Best model automatically selected and stored in `model_metrics.json`

---

### 5. Deployment (Streamlit)

* Interactive UI:
  * Numeric inputs
  * Dropdown selections
* Input validation:
  * Salary must be greater than 0
  * Logical loan constraints
* Outputs:
  * EMI eligibility
  * Maximum monthly EMI

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/harishgundapuUD/EMI_Prediction.git
cd EMI_Prediction
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 📊 Example Inputs

* Age
* Salary
* Employment details
* Expenses
* Loan details
* Credit score

---

## 📈 Output

* ✅ EMI Eligibility (Eligible / Not Eligible)
* 💰 Maximum Monthly EMI

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* MLflow
* Pandas / NumPy
* Streamlit
* Matplotlib / Seaborn (for EDA)

---

## 🚀 Future Improvements

* 📊 Feature importance explanation
* 📈 EMI visualization charts
* 🌐 Cloud deployment (Streamlit Cloud / AWS)
* 🧠 Model explainability (SHAP)

---

## 👨‍💻 Author

**Harish Gundapu**
GitHub: [https://github.com/harishgundapuUD](https://github.com/harishgundapuUD)

---

## 📄 License

This project is open-source and available under the MIT License.

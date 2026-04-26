import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("your_data_file.csv")  # replace with your CSV file path

# Make sure the output directory exists
output_dir = "eda_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Analyze EMI Eligibility Distribution Across Different Lending Scenarios

emi_scenarios = [
    "emi_scenario_e-commerce shopping emi", "emi_scenario_education emi", 
    "emi_scenario_home appliances emi", "emi_scenario_personal loan emi", 
    "emi_scenario_vehicle emi"
]

# Loop through lending scenarios and save plots
for scenario in emi_scenarios:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=scenario, hue="emi_eligibility", data=df)
    plt.title(f"EMI Eligibility Distribution: {scenario}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"emi_eligibility_distribution_{scenario}.png"))
    plt.close()  # Close the plot to avoid memory issues

# 2. Study Correlation Between Financial Variables and Loan Approval Rates

financial_columns = ['monthly_salary', 'total_monthly_expenses', 'debt_to_income_ratio', 
                     'expense_to_income_ratio', 'savings_rate', 'requested_amount', 'requested_tenure']

# Visualizing financial variables vs EMI eligibility (Boxplots)
for column in financial_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="emi_eligibility", y=column, data=df)
    plt.title(f"Distribution of {column} by EMI Eligibility")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"financial_variable_{column}_by_emi_eligibility.png"))
    plt.close()

# Correlation Heatmap for financial variables
financial_data = df[financial_columns]
correlation_matrix = financial_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation between Financial Variables")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "financial_variables_correlation_heatmap.png"))
plt.close()

# 3. Investigate Demographic Patterns and Risk Factor Relationships

# Age Distribution by EMI Eligibility
plt.figure(figsize=(8, 6))
sns.histplot(df[df['emi_eligibility'] == 'eligible']['age'], color="blue", kde=True, label="Eligible")
sns.histplot(df[df['emi_eligibility'] == 'not_eligible']['age'], color="red", kde=True, label="Not Eligible")
plt.legend()
plt.title('Age Distribution by EMI Eligibility')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_distribution_by_emi_eligibility.png"))
plt.close()

# Gender Distribution by EMI Eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="gender_female", hue="emi_eligibility", data=df)
plt.title('EMI Eligibility by Gender')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_gender.png"))
plt.close()

# Marital Status vs EMI Eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="marital_status_married", hue="emi_eligibility", data=df)
plt.title('EMI Eligibility by Marital Status')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_marital_status.png"))
plt.close()

# Family Size and EMI Eligibility
plt.figure(figsize=(8, 6))
sns.boxplot(x="emi_eligibility", y="family_size", data=df)
plt.title('Family Size vs EMI Eligibility')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "family_size_vs_emi_eligibility.png"))
plt.close()

# Existing loans and EMI eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="existing_loans", hue="emi_eligibility", data=df)
plt.title('EMI Eligibility by Existing Loans')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_existing_loans.png"))
plt.close()

# 4. Generate Comprehensive Statistical Summaries and Business Insights

# General statistics summary (save as a CSV)
stat_summary = df.describe()
stat_summary.to_csv(os.path.join(output_dir, "statistical_summary.csv"))

# Additional insights: Gender distribution of applicants
gender_dist = df['gender_female'].value_counts() / len(df) * 100

# Distribution of Loan Eligibility
loan_eligibility_dist = df['emi_eligibility'].value_counts() / len(df) * 100

# Save these insights as a text file
with open(os.path.join(output_dir, "business_insights.txt"), "w") as f:
    f.write("Gender Distribution:\n")
    f.write(str(gender_dist) + "\n\n")
    
    f.write("Loan Eligibility Distribution:\n")
    f.write(str(loan_eligibility_dist) + "\n\n")
    
    f.write("Statistical Summary:\n")
    f.write(str(stat_summary) + "\n")

# Example for saving the MAPE, R2 and other relevant statistics directly
mape_stats = df[['emi_eligibility', 'monthly_salary', 'debt_to_income_ratio']].mean()
with open(os.path.join(output_dir, "mape_and_r2_insights.txt"), "w") as f:
    f.write("MAPE and R2 Insights:\n")
    f.write(str(mape_stats) + "\n")












### Another code variation

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the original and processed datasets
original_data = pd.read_csv("original_data.csv")  # replace with your original CSV file path
processed_data = pd.read_csv("processed_data.csv")  # replace with your processed CSV file path

# Engineered feature columns from the processed data (ensure these are in your config)
engineered_features = [
    "total_monthly_expenses", "disposable_income", "expense_to_income_ratio", 
    "debt_to_income_ratio", "affordability_ratio", "savings_rate", 
    "financial_buffer", "projected_emi_burden"
]

# Remove one-hot encoded columns (you can identify these by excluding specific columns in config)
one_hot_encoded_columns = [
    'gender_female', 'gender_male', 'marital_status_married', 'marital_status_single', 
    'employment_type_government', 'employment_type_private', 'employment_type_self-employed',
    'emi_scenario_e-commerce shopping emi', 'emi_scenario_education emi', 'emi_scenario_home appliances emi', 
    'emi_scenario_personal loan emi', 'emi_scenario_vehicle emi'
]

# Columns to consider for EDA: original data + engineered features from processed data
eda_columns = original_data.drop(columns=one_hot_encoded_columns).columns.tolist() + engineered_features

# Merge original data with engineered features
df = pd.merge(original_data, processed_data[engineered_features], left_index=True, right_index=True)

# Make sure the output directory exists
output_dir = "eda_plots"
os.makedirs(output_dir, exist_ok=True)

# ------------------------
# 1. EMI Eligibility Distribution Across Different Lending Scenarios
emi_scenarios = [
    "emi_scenario_e-commerce shopping emi", "emi_scenario_education emi", 
    "emi_scenario_home appliances emi", "emi_scenario_personal loan emi", 
    "emi_scenario_vehicle emi"
]

# Loop through lending scenarios and save plots
for scenario in emi_scenarios:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=scenario, hue="emi_eligibility", data=df)
    plt.title(f"EMI Eligibility Distribution: {scenario}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"emi_eligibility_distribution_{scenario}.png"))
    plt.close()  # Close the plot to avoid memory issues

# ------------------------
# 2. Study Correlation Between Financial Variables and Loan Approval Rates
financial_columns = [
    'monthly_salary', 'total_monthly_expenses', 'debt_to_income_ratio', 'expense_to_income_ratio', 
    'savings_rate', 'requested_amount', 'requested_tenure'
]

# Visualizing financial variables vs EMI eligibility (Boxplots)
for column in financial_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="emi_eligibility", y=column, data=df)
    plt.title(f"Distribution of {column} by EMI Eligibility")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"financial_variable_{column}_by_emi_eligibility.png"))
    plt.close()

# Correlation Heatmap for financial variables and engineered features
financial_data = df[financial_columns + engineered_features]
correlation_matrix = financial_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation between Financial Variables and Engineered Features")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "financial_and_engineered_features_correlation_heatmap.png"))
plt.close()

# ------------------------
# 3. Investigate Demographic Patterns and Risk Factor Relationships
# Age Distribution by EMI Eligibility
plt.figure(figsize=(8, 6))
sns.histplot(df[df['emi_eligibility'] == 'eligible']['age'], color="blue", kde=True, label="Eligible")
sns.histplot(df[df['emi_eligibility'] == 'not_eligible']['age'], color="red", kde=True, label="Not Eligible")
plt.legend()
plt.title('Age Distribution by EMI Eligibility')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_distribution_by_emi_eligibility.png"))
plt.close()

# Gender Distribution by EMI Eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="gender_female", hue="emi_eligibility", data=df)
plt.title('EMI Eligibility by Gender')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_gender.png"))
plt.close()

# Marital Status vs EMI Eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="marital_status_married", hue="emi_eligibility", data=df)
plt.title('EMI Eligibility by Marital Status')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_marital_status.png"))
plt.close()

# Family Size and EMI Eligibility
plt.figure(figsize=(8, 6))
sns.boxplot(x="emi_eligibility", y="family_size", data=df)
plt.title('Family Size vs EMI Eligibility')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "family_size_vs_emi_eligibility.png"))
plt.close()

# Existing loans and EMI eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="existing_loans", hue="emi_eligibility", data=df)
plt.title('EMI Eligibility by Existing Loans')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_existing_loans.png"))
plt.close()

# ------------------------
# 4. Generate Comprehensive Statistical Summaries and Business Insights
# General statistics summary (save as a CSV)
stat_summary = df.describe()
stat_summary.to_csv(os.path.join(output_dir, "statistical_summary.csv"))

# Additional insights: Gender distribution of applicants
gender_dist = df['gender_female'].value_counts() / len(df) * 100

# Distribution of Loan Eligibility
loan_eligibility_dist = df['emi_eligibility'].value_counts() / len(df) * 100

# Save these insights as a text file
with open(os.path.join(output_dir, "business_insights.txt"), "w") as f:
    f.write("Gender Distribution:\n")
    f.write(str(gender_dist) + "\n\n")
    
    f.write("Loan Eligibility Distribution:\n")
    f.write(str(loan_eligibility_dist) + "\n\n")
    
    f.write("Statistical Summary:\n")
    f.write(str(stat_summary) + "\n")

# Example for saving the MAPE, R2 and other relevant statistics directly
mape_stats = df[['emi_eligibility', 'monthly_salary', 'debt_to_income_ratio']].mean()
with open(os.path.join(output_dir, "mape_and_r2_insights.txt"), "w") as f:
    f.write("MAPE and R2 Insights:\n")
    f.write(str(mape_stats) + "\n")










###### another code variation

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the original and processed datasets
original_data = pd.read_csv("original_data.csv")  # replace with your original CSV file path
processed_data = pd.read_csv("processed_data.csv")  # replace with your processed CSV file path

# Engineered feature columns from the processed data (ensure these are in your config)
engineered_features = [
    "total_monthly_expenses", "disposable_income", "expense_to_income_ratio", 
    "debt_to_income_ratio", "affordability_ratio", "savings_rate", 
    "financial_buffer", "projected_emi_burden"
]

# Remove one-hot encoded columns (you can identify these by excluding specific columns in config)
one_hot_encoded_columns = [
    'gender_female', 'gender_male', 'marital_status_married', 'marital_status_single', 
    'employment_type_government', 'employment_type_private', 'employment_type_self-employed',
    'emi_scenario_e-commerce shopping emi', 'emi_scenario_education emi', 'emi_scenario_home appliances emi', 
    'emi_scenario_personal loan emi', 'emi_scenario_vehicle emi'
]

# Columns to consider for EDA: original data + engineered features from processed data
eda_columns = original_data.drop(columns=one_hot_encoded_columns).columns.tolist() + engineered_features

# Merge original data with engineered features
df = pd.merge(original_data, processed_data[engineered_features], left_index=True, right_index=True)

# Make sure the output directory exists
output_dir = "eda_plots"
os.makedirs(output_dir, exist_ok=True)

# ------------------------
# 1. EMI Eligibility Distribution Across Different Lending Scenarios
emi_scenarios = [
    "emi_scenario_e-commerce shopping emi", "emi_scenario_education emi", 
    "emi_scenario_home appliances emi", "emi_scenario_personal loan emi", 
    "emi_scenario_vehicle emi"
]

# Loop through lending scenarios and save plots
for scenario in emi_scenarios:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=scenario, hue="emi_eligibility", data=df)
    plt.title(f"EMI Eligibility Distribution: {scenario}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"emi_eligibility_distribution_{scenario}.png"))
    plt.close()  # Close the plot to avoid memory issues

# ------------------------
# 2. Study Correlation Between Financial Variables and Loan Approval Rates
financial_columns = [
    'monthly_salary', 'total_monthly_expenses', 'debt_to_income_ratio', 'expense_to_income_ratio', 
    'savings_rate', 'requested_amount', 'requested_tenure', 'max_monthly_emi', 'monthly_rent',
    'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses'
]

# Visualizing financial variables vs EMI eligibility (Boxplots)
for column in financial_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="emi_eligibility", y=column, data=df)
    plt.title(f"Distribution of {column} by EMI Eligibility")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"financial_variable_{column}_by_emi_eligibility.png"))
    plt.close()

# Correlation Heatmap for financial variables and engineered features
financial_data = df[financial_columns + engineered_features]
correlation_matrix = financial_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation between Financial Variables and Engineered Features")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "financial_and_engineered_features_correlation_heatmap.png"))
plt.close()

# ------------------------
# 3. Investigate Demographic Patterns and Risk Factor Relationships
# Age Distribution by EMI Eligibility
plt.figure(figsize=(8, 6))
sns.histplot(df[df['emi_eligibility'] == 'eligible']['age'], color="blue", kde=True, label="Eligible")
sns.histplot(df[df['emi_eligibility'] == 'not_eligible']['age'], color="red", kde=True, label="Not Eligible")
plt.legend()
plt.title('Age Distribution by EMI Eligibility')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_distribution_by_emi_eligibility.png"))
plt.close()

# Gender Distribution by EMI Eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="gender_female", hue="emi_eligibility", data=df)
plt.title('EMI Eligibility by Gender')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_gender.png"))
plt.close()

# Marital Status vs EMI Eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="marital_status_married", hue="emi_eligibility", data=df)
plt.title('EMI Eligibility by Marital Status')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_marital_status.png"))
plt.close()

# Family Size and EMI Eligibility
plt.figure(figsize=(8, 6))
sns.boxplot(x="emi_eligibility", y="family_size", data=df)
plt.title('Family Size vs EMI Eligibility')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "family_size_vs_emi_eligibility.png"))
plt.close()

# Existing loans and EMI eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="existing_loans", hue="emi_eligibility", data=df)
plt.title('EMI Eligibility by Existing Loans')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_existing_loans.png"))
plt.close()

# ------------------------
# 4. Max Monthly EMI Analysis with Other Financial Parameters
# Correlation between max_monthly_emi and other financial parameters
max_emi_columns = [
    'monthly_salary', 'monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
    'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount', 'requested_amount', 
    'requested_tenure', 'debt_to_income_ratio', 'expense_to_income_ratio', 'savings_rate'
]

# Visualize relationship between max_monthly_emi and other financial variables (Boxplots)
for column in max_emi_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="emi_eligibility", y=column, data=df)
    plt.title(f"Distribution of {column} by EMI Eligibility and Max Monthly EMI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"max_emi_vs_{column}.png"))
    plt.close()

# ------------------------
# 5. Generate Comprehensive Statistical Summaries and Business Insights
# General statistics summary (save as a CSV)
stat_summary = df.describe()
stat_summary.to_csv(os.path.join(output_dir, "statistical_summary.csv"))

# Additional insights: Gender distribution of applicants
gender_dist = df['gender_female'].value_counts() / len(df) * 100

# Distribution of Loan Eligibility
loan_eligibility_dist = df['emi_eligibility'].value_counts() / len(df) * 100

# Save these insights as a text file
with open(os.path.join(output_dir, "business_insights.txt"), "w") as f:
    f.write("Gender Distribution:\n")
    f.write(str(gender_dist) + "\n\n")
    
    f.write("Loan Eligibility Distribution:\n")
    f.write(str(loan_eligibility_dist) + "\n\n")
    
    f.write("Statistical Summary:\n")
    f.write(str(stat_summary) + "\n")

# Example for saving the MAPE, R2 and other relevant statistics directly
mape_stats = df[['emi_eligibility', 'monthly_salary', 'debt_to_income_ratio']].mean()
with open(os.path.join(output_dir, "mape_and_r2_insights.txt"), "w") as f:
    f.write("MAPE and R2 Insights:\n")
    f.write(str(mape_stats) + "\n")
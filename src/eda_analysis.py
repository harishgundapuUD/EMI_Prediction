import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the original and processed datasets
original_data = pd.read_csv("dataset/emi_prediction_dataset.csv")  # replace with your original CSV file path
before_encoded_data = pd.read_csv("dataset/before_encoding_data.csv")  # replace with your processed CSV file path
processed_data = pd.read_csv("dataset/processed_data.csv")  # replace with your cleaned CSV file path

# reading the config data
config = {}
with open("utils/config.json", "r") as f:
    config = json.load(f)

if config:
    eda_data = config["eda_analysis"]
    engineered_features = eda_data.get("engineered_features", [])
    output_dir = eda_data.get("base_dir", "eda_analysis")
    financial_columns = eda_data.get("financial_columns", [])
    max_emi_columns = eda_data.get("max_emi_columns", [])

combined_data = pd.concat([before_encoded_data, processed_data[engineered_features]], axis=1)

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# ------------------------
# 1. EMI Eligibility Distribution Across Different Lending Scenarios
# Plotting the EMI eligibility distribution for all EMI scenarios on the same plot
plt.figure(figsize=(12, 6))
sns.countplot(
    x='emi_scenario', 
    hue='emi_eligibility',  # Use hue to differentiate by eligibility
    data=combined_data,  # Use the processed data
    palette="Set2"
)

# Set plot title and labels
plt.title("EMI Eligibility Distribution Across Different Lending Scenarios", fontsize=16)
plt.xlabel("EMI Scenario", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha="right")  # Rotate x labels for better readability
plt.tight_layout()
# plt.show()

# Save the plot
plt.savefig(os.path.join(output_dir, "emi_eligibility_distribution_all_loan_types.png"))
plt.close()  # Close the plot to avoid memory issues

# ------------------------
# 2. Study Correlation Between Financial Variables and Loan Approval Rates

# Visualizing financial variables vs EMI eligibility (Boxplots)
for column in financial_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="emi_eligibility", y=column, data=combined_data)
    plt.title(f"Distribution of {column} by EMI Eligibility")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"financial_variable_{column}_by_emi_eligibility.png"))
    plt.close()

# Correlation Heatmap for financial variables and engineered features
financial_data = combined_data[financial_columns + engineered_features]
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
sns.histplot(combined_data[combined_data['emi_eligibility'] == 'eligible']['age'], color="blue", kde=True, label="Eligible")
sns.histplot(combined_data[combined_data['emi_eligibility'] == 'not_eligible']['age'], color="red", kde=True, label="Not Eligible")
plt.legend()
plt.title('Age Distribution by EMI Eligibility')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_distribution_by_emi_eligibility.png"))
plt.close()

# Gender Distribution by EMI Eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="gender", hue="emi_eligibility", data=combined_data)
plt.title('EMI Eligibility by Gender')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_gender.png"))
plt.close()

# Marital Status vs EMI Eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="marital_status", hue="emi_eligibility", data=combined_data)
plt.title('EMI Eligibility by Marital Status')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_marital_status.png"))
plt.close()

# Family Size and EMI Eligibility
plt.figure(figsize=(8, 6))
sns.boxplot(x="emi_eligibility", y="family_size", data=combined_data)
plt.title('Family Size vs EMI Eligibility')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "family_size_vs_emi_eligibility.png"))
plt.close()

# Existing loans and EMI eligibility
plt.figure(figsize=(8, 6))
sns.countplot(x="existing_loans", hue="emi_eligibility", data=combined_data)
plt.title('EMI Eligibility by Existing Loans')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "emi_eligibility_by_existing_loans.png"))
plt.close()

# ------------------------
# 4. Max Monthly EMI Analysis with Other Financial Parameters
# Correlation between max_monthly_emi and other financial parameters

# Visualize relationship between max_monthly_emi and other financial variables (Boxplots)
for column in max_emi_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="emi_eligibility", y=column, data=combined_data)
    plt.title(f"Distribution of {column} by EMI Eligibility and Max Monthly EMI")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"max_emi_vs_{column}.png"))
    plt.close()

# # ------------------------
# # 5. Generate Comprehensive Statistical Summaries and Business Insights
# # General statistics summary (save as a CSV)
# stat_summary = combined_data.describe()
# stat_summary.to_csv(os.path.join(output_dir, "statistical_summary.csv"))

# # Additional insights: Gender distribution of applicants
# gender_dist = combined_data['gender'].value_counts() / len(combined_data) * 100

# # Distribution of Loan Eligibility
# loan_eligibility_dist = combined_data['emi_eligibility'].value_counts() / len(combined_data) * 100

# # Save these insights as a text file
# with open(os.path.join(output_dir, "business_insights.txt"), "w") as f:
#     f.write("Gender Distribution:\n")
#     f.write(str(gender_dist) + "\n\n")
    
#     f.write("Loan Eligibility Distribution:\n")
#     f.write(str(loan_eligibility_dist) + "\n\n")
    
#     f.write("Statistical Summary:\n")
#     f.write(str(stat_summary) + "\n")

# # Example for saving the MAPE, R2 and other relevant statistics directly
# mape_stats = combined_data[['emi_eligibility', 'monthly_salary', 'debt_to_income_ratio']].mean()
# with open(os.path.join(output_dir, "mape_and_r2_insights.txt"), "w") as f:
#     f.write("MAPE and R2 Insights:\n")
#     f.write(str(mape_stats) + "\n")
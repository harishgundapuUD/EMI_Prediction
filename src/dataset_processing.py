import pandas as pd
import json
import numpy as np

class DatasetLoader:
    def __init__(self, dataset_path, config_path):
        self.dataset_path = dataset_path
        self.config_path = config_path
    
    def load_data(self):
        data = pd.read_csv(self.dataset_path)
        config = {}
        with open(self.config_path, "r") as f:
            config = json.load(f)
        return data, config

class DataCleaning:
    def __init__(self):
        pass

    def clean_numeric_column(self, series):
        return pd.to_numeric(
            series.astype(str)
                .str.rsplit('.', n=1).str[0],
            errors='coerce'
        )

    def clean_categorical(self, series, mapping={}):
        column = series.name
        s = (
            series.astype(str)
                .str.strip()
                .str.lower()
                .replace("nan", pd.NA)   # default handling
        )
        
        if mapping and column in mapping:
            s = s.replace(mapping[column])
        
        return s

class DataPreprocessor:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.processed_data_path = self.config.get("processed_data_path", "dataset/processed_data.csv")
        self.before_encoding_path = self.config.get("before_encoding_path", "dataset/before_encoding_data.csv")
        self.cleaner = DataCleaning()
    
    def preprocess(self, testing=False):
        numerical_columns = self.config.get("numerical_columns")
        categorical_columns = self.config.get("categorical_columns")

        if testing:
            numerical_columns = [i for i in numerical_columns if i not in self.config.get("drop_columns", [])]
            categorical_columns = [i for i in categorical_columns if i not in self.config.get("drop_columns", [])]

        self.data[numerical_columns] = self.data[numerical_columns].apply(self.cleaner.clean_numeric_column)
        self.data[categorical_columns] = self.data[categorical_columns].apply(lambda col: self.cleaner.clean_categorical(col, mapping=self.config["mapping"]))
        self.data = self.data[self.data["credit_score"].notna() & (self.data["credit_score"] != 0)]
       
        # Apply limits
        mask = pd.Series(True, index=self.data.index)
        for col, (min_val, max_val) in self.config["limits"].items():
            mask &= self.data[col].between(min_val, max_val, inclusive="both")
        self.data = self.data[mask]
        
        # remove rows with missing montly_salary or requested_amount or requested_tenure
        self.data = self.data.dropna(subset=["monthly_salary", "requested_amount", "requested_tenure"])
        self.data = self.data.drop(self.config.get("less_important_cols", []), axis=1)

        # remove rows where the house type is rented and montly rent is missing or <= 0
        # self.data = self.data[
        #                         self.data["monthly_rent"].notna()
        #                         & (self.data["monthly_rent"] > 0)
        #                         & self.data["house_type"].notna()
        #                     ]
        self.data = self.data[
                                ~(
                                    self.data["house_type"].eq("rented")
                                    & self.data["monthly_rent"].isna()
                                )
                            ]
        # self.data["monthly_rent"] = self.data["monthly_rent"].fillna(self.config.get("default_values", {}).get("monthly_rent", 0))

        # apply the default value for missing column values
        for col, default_val in self.config.get("default_values", {}).items():
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(default_val)
        if not testing:
            self.data.to_csv(self.before_encoding_path, index=False)

        # apply one-hot encoding for categorical columns
        self.data = pd.get_dummies(self.data, columns=self.config.get("one_hot_encode_cols", []), drop_first=False)

        # apply label encoding for categorical columns
        self.data = self.data.apply(
                                        lambda col: col.map(
                                            {k: v for v, k in enumerate(
                                                self.config["label_encoding_order"].get(col.name, [])
                                            )}
                                        ) if col.name in self.config.get("label_encode_cols", []) else col
                                    )

    def create_financial_features(self, testing=False):
        df = self.data.copy()
        
        # ---- 1. Total Monthly Expenses ----
        df["total_monthly_expenses"] = df[self.config.get("expense_cols")].sum(axis=1)
        
        # ---- 2. Disposable Income ----
        df["disposable_income"] = df["monthly_salary"] - df["total_monthly_expenses"]
        
        # ---- 3. Expense to Income Ratio ----
        df["expense_to_income_ratio"] = df["total_monthly_expenses"] / df["monthly_salary"]
        
        # ---- 4. Debt to Income Ratio ----
        df["debt_to_income_ratio"] = df["current_emi_amount"] / df["monthly_salary"]
        
        # ---- 5. Affordability Ratio (loan burden estimate) ----
        df["affordability_ratio"] = (df["requested_amount"] / (df["monthly_salary"] * df["requested_tenure"]))
        
        # ---- 6. Savings Rate ----
        df["savings_rate"] = df["disposable_income"] / df["monthly_salary"]
        
        # ---- 7. Financial buffer (liquidity strength) ----
        df["financial_buffer"] = df["bank_balance"] + df["emergency_fund"]
        
        # ---- 8. EMI burden after loan request (stress proxy) ----
        df["projected_emi_burden"] = (df["current_emi_amount"] + 
                                      (df["requested_amount"] / df["requested_tenure"])) / df["monthly_salary"]
        
        # ---- clean inf/nan ----
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if not testing:
            df.to_csv(self.processed_data_path, index=False)
            train_columns = {"train_columns": df.columns.tolist()}
            with open("utils/train_columns.json", "w") as f:
                json.dump(train_columns, f, indent=4)
        return df

data_loader = DatasetLoader(dataset_path="dataset/emi_prediction_dataset.csv", config_path="utils/config.json")
data, config = data_loader.load_data()
preprocessor = DataPreprocessor(data=data, config=config)
preprocessor.preprocess()
processed_data = preprocessor.create_financial_features()


'''
# this has to be done while testing the model from the streamlit app

test_encoded = pd.get_dummies(
    test_df,
    columns=self.config.get("one_hot_encode_cols", []),
    drop_first=False
)

test_encoded = test_encoded.reindex(columns=train_columns, fill_value=0)
'''
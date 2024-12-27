import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import save_object  # Ensure this utility function is defined in src.utils
from src.logger import logging  # Ensure logging is properly set up

@dataclass
class RegressionDataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifact', "regression", "preprocessor.pkl.gz")

class OutlierIQRWithImputationTransformer(BaseEstimator, TransformerMixin):
    """
    Handles missing values and outliers in specified columns using imputation
    and IQR-based clipping.

    Parameters:
    - columns (list): List of columns to handle.
    - imputer_strategy (str): Strategy for imputation, e.g., 'median'.
    """
    def __init__(self, columns, imputer_strategy='median'):
        self.columns = columns
        self.imputer_strategy = imputer_strategy
        self.imputer = SimpleImputer(strategy=self.imputer_strategy)
        self.lower_thresholds = {}
        self.upper_thresholds = {}

    def fit(self, X, y=None):
        """
        Fits the imputer on the data and calculates the thresholds for outliers based on IQR.
        """
        # Fit the imputer on the selected columns
        self.imputer.fit(X[self.columns])

        # Calculate and save the lower and upper thresholds for IQR-based clipping
        for column in self.columns:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_threshold = Q1 - 1.5 * IQR
            upper_threshold = Q3 + 1.5 * IQR

            self.lower_thresholds[column] = lower_threshold
            self.upper_thresholds[column] = upper_threshold
        
        return self

    def transform(self, X):
        """
        Imputes missing values and applies outlier clipping.
        """
        df = X.copy()

        # Impute missing values for the specified columns
        df[self.columns] = self.imputer.transform(df[self.columns])

        # Apply the IQR-based clipping for outliers
        for column in self.columns:
            lower_threshold = self.lower_thresholds[column]
            upper_threshold = self.upper_thresholds[column]
            df[column] = df[column].clip(lower=lower_threshold, upper=upper_threshold)

        return df

class RegressionDataTransformation:
    def __init__(self):
        self.data_transformation_config = RegressionDataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define columns
            numeric_columns = ['quantity tons', 'country', 'application', 'thickness', 'width',
                               'product_ref', 'item_day', 'item_month', 'item_year',
                               'delivery_day', 'delivery_month', 'delivery_year']
            
            discrete_categorical_columns = ['status', 'item type']
            
            # Numeric pipeline
            num_pipeline = Pipeline([
                ("outlier_handler", OutlierIQRWithImputationTransformer(columns=numeric_columns, imputer_strategy='median')),
                ('scaler', RobustScaler())
            ])

            # Discrete Categorical pipeline (One-Hot Encoding)
            discrete_cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # Combine all pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numeric_columns),
                ("discrete_cat", discrete_cat_pipeline, discrete_categorical_columns),
            ])

            return preprocessor
        
        except Exception as e:
            logging.error(f"Error in creating transformer: {str(e)}")
            return None

    def initiate_data_transformer(self, train_path, test_path):
        try:
            # Read data
            train_df = pd.read_excel(train_path)
            test_df = pd.read_excel(test_path)
            logging.info("Read train and test data completed")

            # Ensure column names are clean
            train_df.columns = train_df.columns.str.strip().str.lower()
            test_df.columns = test_df.columns.str.strip().str.lower()

            target_column_name = 'selling_price'

            # Check if target column exists
            if target_column_name not in train_df.columns:
                raise Exception(f"Target column '{target_column_name}' not found in training data")
            if target_column_name not in test_df.columns:
                raise Exception(f"Target column '{target_column_name}' not found in testing data")

            # Apply log transformation to target column if necessary
            train_df[target_column_name] = np.log1p(train_df[target_column_name])
            test_df[target_column_name] = np.log1p(test_df[target_column_name])

            # Separate input and target columns
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Get preprocessor and apply transformations
            preprocessing_obj = self.get_data_transformer_object()
            if preprocessing_obj is None:
                raise Exception("Preprocessing object creation failed.")
            
            logging.info("Applying preprocessing on training and testing data.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed data with target columns
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]

            # Save the preprocessing object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)
            logging.info(f"Saved preprocessing object at {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr
        
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
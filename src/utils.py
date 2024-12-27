import os
import sys
import pickle
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import warnings
from src.exception import CustomException
from src.logger import logging
import gzip

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output


def save_object(file_path, obj):
    """Save an object to a compressed pickle file using gzip."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with gzip.open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load an object from a compressed pickle file using gzip."""
    try:
        with gzip.open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param, task_type='regression'):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training and evaluating model: {model_name}")
            
            # Get hyperparameters for the model
            params = param.get(model_name, {})
            
            # Perform GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, params, cv=3, scoring='accuracy' if task_type == 'classification' else 'r2', n_jobs=-1)
            gs.fit(X_train, y_train)

            # Log best parameters
            best_params = gs.best_params_
            logging.info(f"Best parameters for {model_name}: {best_params}")

            # Set the best parameters to the model
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            # Predictions for training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics based on task type (regression or classification)
            if task_type == 'regression':
                train_r2 = r2_score(y_train, y_train_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                train_mse = mean_squared_error(y_train, y_train_pred)

                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)

                # Log regression metrics
                logging.info(f"Metrics for {model_name}:")
                logging.info(f"  Training R²: {train_r2:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}")
                logging.info(f"  Testing  R²: {test_r2:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}")
                logging.info("-" * 50)

                # Store the regression metrics
                report[model_name] = {
                    "test_r2": test_r2,
                    "test_mae": test_mae,
                    "test_mse": test_mse,
                    "best_params": best_params,
                }

            elif task_type == 'classification':
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_precision = precision_score(y_train, y_train_pred, average='weighted')
                train_recall = recall_score(y_train, y_train_pred, average='weighted')
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')

                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(y_test, y_test_pred, average='weighted')
                test_recall = recall_score(y_test, y_test_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')

                # Log classification metrics
                logging.info(f"Metrics for {model_name}:")
                logging.info(f"  Training Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")
                logging.info(f"  Testing  Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")
                logging.info("-" * 50)

                # Store the classification metrics
                report[model_name] = {
                    "test_accuracy": test_accuracy,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1,
                    "best_params": best_params,
                }

        return report

    except Exception as e:
        raise CustomException(e, sys)


class RawData:

    def __init__(self,file_path):
        self.data = pd.read_excel(file_path)

    def regression_data(self):
        df = self.data.copy()
        df.drop(df[df['selling_price'] < 1].index, inplace=True)
        df.dropna(subset=['selling_price'], inplace=True)
        df['item_date'] = pd.to_datetime(df['item_date'],format='%Y%m%d',errors='coerce')
        df['delivery date'] = pd.to_datetime(df['delivery date'],format = '%Y%m%d',errors='coerce')
        df['item_day'] = df['item_date'].dt.day
        df['item_month'] = df['item_date'].dt.month
        df['item_year'] = df['item_date'].dt.year
        df['delivery_day'] = df['delivery date'].dt.day
        df['delivery_month'] = df['delivery date'].dt.month
        df['delivery_year'] = df['delivery date'].dt.year
        df['quantity tons'] = pd.to_numeric(df['quantity tons'],errors='coerce')
        df.drop(columns = ['material_ref','id','customer','delivery date','item_date'],inplace = True)
        
        return df
    
    def classification_data(self):
        df = self.data.copy()
        df = df[df['status'].isin(['Won', 'Lost'])]
        df['status'] = df['status'].replace({'Won': 0, 'Lost': 1})
        df['item_date'] = pd.to_datetime(df['item_date'],format='%Y%m%d',errors='coerce')
        df['delivery date'] = pd.to_datetime(df['delivery date'],format = '%Y%m%d',errors='coerce')
        df['item_day'] = df['item_date'].dt.day
        df['item_month'] = df['item_date'].dt.month
        df['item_year'] = df['item_date'].dt.year
        df['delivery_day'] = df['delivery date'].dt.day
        df['delivery_month'] = df['delivery date'].dt.month
        df['delivery_year'] = df['delivery date'].dt.year
        df['quantity tons'] = pd.to_numeric(df['quantity tons'],errors='coerce')
        df.drop(columns = ['material_ref','id','customer','delivery date','item_date'],inplace = True)
        
        return df


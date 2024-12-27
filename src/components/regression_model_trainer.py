import os
import sys
from dataclasses import dataclass
from sklearn.svm import SVR
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class RegressionModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","regression","model.pkl.gz")

class RegressionModelTrainer:
    def __init__(self):
        self.model_trainer_config = RegressionModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                # Linear Models
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),

                # Tree-based Models
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),

                # KNN
                "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
            }

            params = {
                # Linear Models
                "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso Regression": {"alpha": [0.1, 1.0, 10.0]},

                # Tree-based Models
                "Decision Tree Regressor": {
                    "max_depth": [2, 4, 6],
                    "min_samples_split": [2, 5],
                },
                "Random Forest Regressor": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5],
                },
                "Gradient Boosting Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                },

                # KNN
                "K-Nearest Neighbors Regressor": {
                    "n_neighbors": [3, 5],
                    "weights": ["uniform", "distance"],
                },
            }

            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
                task_type="regression",
            )
            
            # Find the best model
            best_model_name = max(model_report, key=lambda x: model_report[x]["test_r2"])
            best_model_score = model_report[best_model_name]["test_r2"]
            best_model_params = model_report[best_model_name]["best_params"]

            # Log the best model details
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Test R²: {best_model_score}")
            logging.info(f"Best Parameters: {best_model_params}")

            # Raise an exception if no model is satisfactory
            if best_model_score < 0.6:
                raise CustomException("No satisfactory model found with R² > 0.6")

            # Save the best model
            best_model = models[best_model_name]
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Test the best model
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
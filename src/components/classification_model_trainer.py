import os
import sys
from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ClassificationModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "classification", "model.pkl.gz")

class ClassificationModelTrainer:
    def __init__(self):
        self.model_trainer_config = ClassificationModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define classifier models
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                
            }

            # Define hyperparameters for tuning
            params = {
                "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
                "Decision Tree Classifier": {
                    "max_depth": [2, 4, 6],
                    "min_samples_split": [2, 5],
                },
                "Random Forest Classifier": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5],
                },
                "Gradient Boosting Classifier": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                },
                "AdaBoost Classifier": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
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
                task_type="classification",  # Use classification metrics
            )

            # Find the best model
            best_model_name = max(model_report, key=lambda x: model_report[x]["test_accuracy"])
            best_model_score = model_report[best_model_name]["test_accuracy"]
            best_model_params = model_report[best_model_name]["best_params"]

            # Log the best model details
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Test Accuracy: {best_model_score}")
            logging.info(f"Best Parameters: {best_model_params}")

            # Raise an exception if no model is satisfactory
            if best_model_score < 0.6:
                raise CustomException("No satisfactory model found with accuracy > 0.6")

            # Save the best model
            best_model = models[best_model_name]
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Test the best model
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
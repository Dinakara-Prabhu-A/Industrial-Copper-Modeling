# Regression
from src.components.regression_data_ingestion import RegressionDataIngestion
from src.components.regression_data_transformation import RegressionDataTransformation
from src.components.regression_model_trainer import RegressionModelTrainer
# Classification
from src.components.classification_data_ingestion import classificationDataIngestion
from src.components.classification_data_transformation import ClassificationDataTransformation
from src.components.classification_model_trainer import ClassificationModelTrainer

if __name__ == '__main__':
    regression_obj = RegressionDataIngestion()
    regression_train_data, regression_test_data = regression_obj.initiate_data_ingestion()
    regression_data_transformation_obj = RegressionDataTransformation()
    regression_train_data, regression_test_data = regression_data_transformation_obj.initiate_data_transformer(regression_train_data, regression_test_data)
    regression_model_trainer_obj = RegressionModelTrainer()
    regression_result = regression_model_trainer_obj.initiate_model_trainer(regression_train_data, regression_test_data)
    print("Final Regression Model Metrics:", regression_result)

    classification_obj = classificationDataIngestion()
    classification_train_data, classification_test_data = classification_obj.initiate_data_ingestion()
    classification_data_transformation_obj = ClassificationDataTransformation()
    classification_train_data, classification_test_data = classification_data_transformation_obj.initiate_data_transformer(classification_train_data, classification_test_data)
    classification_model_trainer_obj = ClassificationModelTrainer()
    classification_result = classification_model_trainer_obj.initiate_model_trainer(classification_train_data, classification_test_data)
    print("Final Classification Model Metrics:", classification_result)
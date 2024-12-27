import sys
import pandas as pd
from src.exception import CustomException
from sklearn.exceptions import NotFittedError
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        self.reg_model_path = 'artifact/regression/model.pkl'
        self.reg_preprocessor_path = 'artifact/regression/preprocessor.pkl'
        self.class_model_path = 'artifact/classification/model.pkl'
        self.class_preprocessor_path = 'artifact/classification/preprocessor.pkl'

    def reg_predict(self,features):
        try:
            reg_model = load_object(self.reg_model_path)
            reg_preprocessor = load_object(self.reg_preprocessor_path)
            # Validate that the model has been fitted
            if not hasattr(reg_model, "predict"):
                raise NotFittedError("The loaded model is not fitted. Train and save the model before using it.")
            # Scale the features and make predictions
            data_scaled = reg_preprocessor.transform(features)
            preds = reg_model.predict(data_scaled)
            return preds
        except NotFittedError as e:
            raise CustomException(f"Model Error: {e}", sys)
        except Exception as e:
            raise CustomException(e, sys)
        
    def class_predict(self,features):
        try:
            class_model = load_object(self.class_model_path)
            class_preprocessor = load_object(self.class_preprocessor_path)
            # Validate that the model has been fitted
            if not hasattr(class_model, "predict"):
                raise NotFittedError("The loaded model is not fitted. Train and save the model before using it.")
            # Scale the features and make predictions
            data_scaled = class_preprocessor.transform(features)
            preds = class_model.predict(data_scaled)
            return preds
        except NotFittedError as e:
            raise CustomException(f"Model Error: {e}", sys)
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 quantity_tons: int,
                 country: int,
                 item_type: str,
                 application: str,
                 thickness: int,
                 width: int,
                 product_ref: int,
                 item_day: int,
                 item_month: int,
                 item_year: int,
                 delivery_day: int,
                 delivery_month: int,
                 delivery_year: int,
                 status: str = None,  # Optional for classification
                 selling_price: int = None  # Optional for classification
                 ):
        # Initialize common fields
        self.quantity_tons = quantity_tons
        self.country = country
        self.item_type = item_type
        self.application = application
        self.thickness = thickness
        self.width = width
        self.product_ref = product_ref
        self.item_day = item_day
        self.item_month = item_month
        self.item_year = item_year
        self.delivery_day = delivery_day
        self.delivery_month = delivery_month
        self.delivery_year = delivery_year
        
        # Specific fields for classification or regression
        self.status = status
        self.selling_price = selling_price

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary of common fields
            custom_data_input_dict = {
                "quantity tons": [self.quantity_tons],
                "country": [self.country],
                "item type": [self.item_type],
                "application": [self.application],
                "thickness": [self.thickness],
                "width": [self.width],
                "product_ref": [self.product_ref],
                "item_day": [self.item_day],
                "item_month": [self.item_month],
                "item_year": [self.item_year],
                "delivery_day": [self.delivery_day],
                "delivery_month": [self.delivery_month],
                "delivery_year": [self.delivery_year]
            }

            # Add specific fields based on whether it's classification or regression
            if self.status is not None:  # For classification
                custom_data_input_dict["status"] = [self.status]
            if self.selling_price is not None:  # For regression
                custom_data_input_dict["selling_price"] = [self.selling_price]

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
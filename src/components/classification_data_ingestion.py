import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import RawData
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class classificationDataIngestionConfig:
    train_data_path: str=os.path.join('artifact','classification','train.xlsx')
    test_data_path: str=os.path.join('artifact','classification','test.xlsx')
    raw_data_path: str=os.path.join('artifact','classification','data.xlsx')

class classificationDataIngestion:
    def __init__(self):
        
        self.ingenstion_config = classificationDataIngestionConfig()

    

    def initiate_data_ingestion(self):
        logging.info(f'{"-"*10}classification Data Ingestion initiated{"-"*10}')
        logging.info("Entered the data ingestion method or components")
        try:
            
            data = RawData('notebook/raw_data/data.xlsx')
            df = data.classification_data() 
            logging.info("Read the Dataset as DataFrame")


            # Save raw data after corrections
            os.makedirs(os.path.dirname(self.ingenstion_config.train_data_path), exist_ok=True)
            df.to_excel(self.ingenstion_config.raw_data_path, index=False, header=True)

            # Train-test split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_excel(self.ingenstion_config.train_data_path, index=False, header=True)
            test_set.to_excel(self.ingenstion_config.test_data_path, index=False, header=True)

            logging.info("Train and Test datasets saved successfully")
            return (
                self.ingenstion_config.train_data_path,
                self.ingenstion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
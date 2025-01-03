# Industry Copper Modeling: Sales Pricing and Status Prediction 

## Project Overview

This project aims to solve two key problems in the copper industry using machine learning models:

1. **Sales Pricing Prediction**: Accurately predict the `Selling_Price` (continuous variable) by addressing challenges such as skewed data and outliers, which can negatively impact manual pricing predictions.
  
2. **Status Classification**: Classify the status of leads as either `WON` or `LOST` to help evaluate the likelihood of a lead converting into a customer.

The project leverages **machine learning regression** for pricing prediction and **classification models** for lead status prediction. Additionally, the project is equipped with a **Streamlit web application** for user interaction and **CI/CD pipeline** for continuous integration, delivery, and deployment. Hyperparameter tuning is applied to select the best models based on their performance (R² score for regression and F1-score for classification).

## Project Design

### Data Preprocessing
Data preprocessing is a crucial part of the project to ensure the models are trained on clean and consistent data:
- **Handling Missing Values**: Imputed missing values in categorical columns with the mode and numeric columns with the median.
- **Outlier Detection**: Identified and handled outliers using statistical methods to avoid skewing the models.
- **Feature Scaling**: Applied `StandardScaler` for numeric features to standardize the data, and `LabelEncoder`/`OneHotEncoder` for categorical features to convert them into a suitable format.
- **Train-Test Split**: Split the dataset into training and testing sets to evaluate model performance reliably.

### Regression Model (Selling Price Prediction)
A **regression model** predicts the `Selling_Price`. Multiple algorithms, such as **Linear Regression**, **Random Forest**, **Ridge Regression**, and others, were trained. The best model was selected based on the **R² score**, with **hyperparameter tuning** applied to optimize the model's performance.

### Classification Model (Lead Status Prediction)
A **classification model** predicts the `Status` as `WON` or `LOST`. Algorithms such as **Logistic Regression**, **Random Forest Classifier**, and **XGBoost** were evaluated, and the best model was chosen based on the **F1-score**. Data preprocessing for this task includes encoding the `Status` variable to binary values.

### Hyperparameter Tuning
Both regression and classification models underwent **hyperparameter tuning** using techniques like **GridSearchCV** and **RandomizedSearchCV** to identify the optimal set of parameters for improving the model performance.

### Streamlit Web Application
A **Streamlit web application** was developed to allow users to interact with the model:
- For **regression**, users can input feature values and get the predicted `Selling_Price`.
- For **classification**, users can input feature values and get the predicted `Status` (either `WON` or `LOST`).

### CI/CD Pipeline
The project is integrated with a **CI/CD pipeline** using **GitHub Actions**:
- **Continuous Integration (CI)**: Ensures the code passes unit tests, is linted, and meets coding standards before being merged into the main branch.
- **Continuous Delivery (CD)**: Automates the process of building, testing, and deploying the Docker container to AWS. This ensures that the latest version of the model is deployed seamlessly.

### Docker and AWS Deployment
The project uses **Docker** for containerizing the machine learning models, making it easy to deploy and manage in any environment. The Docker container is deployed on **Amazon Web Services (AWS)** using **Elastic Container Registry (ECR)** and **Elastic Container Service (ECS)**.

## Project Directory Structure

```bash

Industrial-copper-modeling
├── /src
│   ├── /components
│   │   ├── regression_model_trainer.py  # Contains the logic for training regression models.
│   │   ├── classification_model_trainer.py  # Contains the logic for training classification models.
│   │   └── predict_pipeline.py  # Logic for making predictions with the trained models.
│   ├── /exception
│   │   └── CustomException.py  # Handles exceptions and error management.
│   ├── /logger
│   │   └── logging.py  # Configures logging for the project.
│   ├── /utils
│   │   └── utils.py  # Contains utility functions for model saving, loading, and evaluation.
│   └── /pipeline
│       └── predict_pipeline.py  # The final prediction pipeline that integrates both models for inference.
├── /.github
│   └── /workflows
│       └── main.yaml  # GitHub Actions configuration for CI/CD pipeline.
├── Dockerfile  # Contains Docker build instructions for containerizing the application.
├── main.yaml   # GitHub Actions configuration file for CI/CD pipeline.
├── README.md   # Project documentation.
├── requirements.txt  # Python dependencies for the project.
└── app.py  # The Streamlit application code for serving predictions.
```

## Steps to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Dinakara-Prabhu-A/Industrial-Copper-Modeling.git
cd copper-sales-lead-classification
```

### 2. Install Dependencies

Make sure you have Python 3.8 or higher installed. Then, install the necessary Python dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### 3. Configure AWS Credentials 

Before running the application, ensure that you have configured your AWS credentials. You can do this by setting the following environment variables:

```bash
export AWS_ACCESS_KEY_ID=your-aws-access-key-id
export AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
export AWS_REGION=your-aws-region
```

### 4. Run the Streamlit Application

To start the Streamlit application locally, use the following command:

```bash
streamlit run app.py
```

### 5. Docker Setup 

If you prefer to run the project inside a Docker container, follow these steps:

 - Build the Docker image:

   ```bash
   docker build -t copper-sales-lead-classification .
 - Run the Docker container:

   ```bash
   docker run -p 8501:8501 copper-sales-lead-classification

This will start the Streamlit app inside the Docker container, accessible at `http://localhost:8501`.

### 6. CI/CD Setup (GitHub Actions)

The project is set up with a CI/CD pipeline using GitHub Actions. The pipeline automates the process of building, testing, and deploying the application. To use the CI/CD pipeline:

1. Push your changes to the `main` branch.
2. GitHub Actions will automatically trigger the pipeline defined in the `.github/workflows/main.yaml` file.
3. The pipeline will:
   - Run unit tests
   - Build the Docker image and push it to Amazon ECR
   - Deploy the application to AWS ECS
4. Ensure that your repository contains the required secrets (AWS credentials, ECR repository details) in the GitHub Secrets settings.

### 7. Model Hyperparameter Tuning

The project includes hyperparameter tuning for both the regression and classification models using cross-validation. The best models are selected based on their performance metrics:
- **Regression Model**: R² score
- **Classification Model**: F1 Score

These models are tuned to ensure the best performance for predicting the `Selling_Price` and classifying the `Status` (WON or LOST).

### 8. Model Deployment

The trained models are deployed as part of the Docker container, which allows for serving predictions via the Streamlit app. The models are loaded at runtime from the Docker container, and predictions are made using the latest trained models.







# Adey_FraudDetection
Project Overview
This project focuses on building robust and accurate fraud detection models to identify fraudulent transactions in e-commerce and banking. Adey Innovations Inc., a leading fintech company, seeks solutions that enhance transaction security and protect against financial loss. The project involves designing and deploying models capable of real-time monitoring and fraud detection using advanced machine learning techniques.

Business Need
Accurate fraud detection is critical for improving the security of online transactions. This project addresses the following business objectives:

Enhanced Security: By leveraging machine learning models, Adey Innovations Inc. aims to detect fraudulent activity efficiently, reducing financial risks and reinforcing customer trust.
Real-time Detection and Monitoring: Implementing a fraud detection model that allows continuous monitoring, timely intervention, and fast reporting.
Geolocation and Pattern Recognition: Integrating analysis based on transaction locations and user behavior patterns to refine detection accuracy.
Project Goals
Data Analysis and Preprocessing: Clean and preprocess transaction datasets.
Feature Engineering: Extract features to identify fraud patterns.
Model Building and Training: Train and evaluate multiple machine learning models for both e-commerce and bank credit transactions.
Model Explainability: Implement SHAP and LIME for model interpretability.
Model Deployment: Deploy models for real-time fraud detection with Flask and Docker.
Dashboard Development: Create a dashboard using Flask and Dash to visualize insights and metrics.
Data Sources
1. Fraud_Data.csv - E-commerce Transaction Data
Attributes:
user_id: Unique identifier for users.
signup_time, purchase_time: Timestamps.
purchase_value: Transaction value.
device_id: Device ID used for transaction.
source: User acquisition source (e.g., SEO, Ads).
browser: Browser used.
sex, age: User demographics.
ip_address: Transaction IP address.
class: Target variable, 1 for fraud, 0 for non-fraud.
2. IpAddress_to_Country.csv - IP to Country Mapping
Attributes:
lower_bound_ip_address, upper_bound_ip_address: IP address range.
country: Country for IP range.
3. creditcard.csv - Bank Transaction Data
Attributes:
Time: Time elapsed from the first transaction.
V1 to V28: Anonymized features from PCA.
Amount: Transaction amount.
Class: Target variable, 1 for fraud, 0 for non-fraud.
Learning Outcomes
Skills:

Deploying models with Flask and containerizing with Docker.
Building REST APIs for model interaction.
Developing deployment pipelines and building dashboards with Dash.
Knowledge:

Principles of model deployment, API best practices, containerization.
Real-time serving techniques, API security, and model monitoring.
Communication:

Report complex statistical insights effectively.
Tasks & Deliverables
Task 1 - Data Analysis and Preprocessing
Handle Missing Values: Impute or drop as needed.
Data Cleaning: Remove duplicates, correct data types.
Exploratory Data Analysis: Conduct univariate and bivariate analysis.
Merge Datasets for Geolocation Analysis: Merge Fraud_Data.csv and IpAddress_to_Country.csv.
Feature Engineering: Create transaction frequency, velocity, time-based features.
Normalization and Scaling: Prepare data for modeling.
Encode Categorical Features
Task 2 - Model Building and Training
Data Preparation: Separate features and target, perform train-test split.
Model Selection: Compare Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, MLP, CNN, RNN, and LSTM.
Model Training and Evaluation: Train on creditcard.csv and Fraud_Data.csv.
MLOps Steps: Track experiments and version models with MLflow.
Task 3 - Model Explainability
Use SHAP and LIME for model interpretability:

SHAP: Create Summary, Force, and Dependence plots.
LIME: Generate feature importance plots for individual predictions.
Task 4 - Model Deployment and API Development
Flask API Development:
Create a Flask app for model serving.
Define API endpoints and test them.
Dockerization:
Dockerize the Flask app, exposing port 5000.
Logging:
Integrate logging for monitoring requests, errors, and fraud predictions.
Task 5 - Dashboard Development with Flask and Dash
Dashboard Insights:
Summary of total transactions, fraud cases, and percentages.
Time-series charts for fraud detection trends.
Geographical fraud analysis and device/browser comparisons.


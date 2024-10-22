# Adey_FraudDetection
Task 1 - Data Analysis and Preprocessing
1. Handle Missing Values
Impute or drop missing values: Ensure that any missing values in the dataset are either filled using strategies like mean, median, or mode imputation or dropped to maintain data integrity.
2. Data Cleaning
Remove duplicates: Identify and remove duplicate records to ensure data quality.
Correct data types: Ensure that each column has the appropriate data type, such as converting columns with date data into datetime format or ensuring that categorical variables are encoded correctly.
3. Exploratory Data Analysis (EDA)
Univariate analysis: Examine each variable individually to understand its distribution and characteristics. This can include measures of central tendency, dispersion, and the distribution of data.
Bivariate analysis: Analyze the relationships between two variables, looking for potential correlations, patterns, or associations that could be relevant to the model.
4. Merge Datasets for Geolocation Analysis
Convert IP addresses to integer format: Convert IP addresses to a numerical format to facilitate merging and analysis.
Merge Fraud_Data.csv with IpAddress_to_Country.csv: Merge the transaction dataset with the geolocation dataset to enrich the data with geographical insights.
5. Feature Engineering
Transaction frequency and velocity: Create new features from the transaction data, such as calculating the frequency of transactions and their velocity over time.
Time-based features: Extract useful time-based features from the timestamp data, including:
hour_of_day: The specific hour when the transaction occurred.
day_of_week: The day of the week the transaction was made to explore potential temporal patterns.
6. Normalization and Scaling
Normalize and scale the features to ensure that they are on the same scale, which can help improve model performance, especially for distance-based algorithms.
7. Encode Categorical Features
Convert categorical variables into numerical representations using techniques such as one-hot encoding or label encoding, depending on the number of categories and the model requirements.
Task 2 - Model Building and Training
1. Data Preparation
Feature and target separation: Split the dataset into features (independent variables) and the target (dependent variable), such as separating 'Class' in the credit card dataset or 'class' in the fraud-data dataset.
Train-test split: Divide the dataset into training and testing sets, ensuring that the model can be evaluated on unseen data.
2. Model Selection
Use several machine learning models to compare their performance, such as:
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
Multi-Layer Perceptron (MLP)
Convolutional Neural Network (CNN)
Recurrent Neural Network (RNN)
Long Short-Term Memory (LSTM)
3. Model Training and Evaluation
Train and evaluate the selected models on both the credit card dataset and the fraud-data dataset, using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC to assess performance.
4. MLOps Steps
Versioning and experiment tracking: Use tools like MLflow to track experiments, including logging parameters, performance metrics, and model versions. This ensures proper documentation and reproducibility of results.
Task 3 - Model Explainability
Model explainability is essential for ensuring that the machine learning models are interpretable, trustworthy, and actionable. This is particularly crucial in fraud detection, where understanding the decision-making process is important for stakeholders.

1. Using SHAP for Explainability
SHAP values provide a global and local interpretation of the model by quantifying the contribution of each feature to the predictions. It helps in understanding which features are most important in driving the model's decisions.

SHAP Plots:
Summary Plot: Provides an overview of the top features and their impact across all predictions.
Force Plot: Illustrates the contribution of individual features to a specific prediction, showing how features push the prediction higher or lower.
Dependence Plot: Visualizes the relationship between a particular feature and the model’s output, highlighting interactions between features.
2. Using LIME for Explainability
LIME explains individual predictions by approximating the model’s behavior locally using interpretable models. It helps in explaining why a specific prediction was made, focusing on individual instances.

LIME Plots:
Feature Importance Plot: Shows the most important features for a specific prediction, giving insights into what influenced the model's decision.

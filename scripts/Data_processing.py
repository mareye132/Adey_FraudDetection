# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set display options
pd.set_option('display.max_columns', None)

# Function to preprocess data
def preprocess_data():
    # Load datasets
    creditcard_df = pd.read_csv(r'C:/Users/user/Desktop/Github/Adey_FraudDetection/data/creditcard.csv')
    fraud_df = pd.read_csv(r'C:/Users/user/Desktop/Github/Adey_FraudDetection/data/Fraud_Data.csv')
    ip_df = pd.read_csv(r'C:/Users/user/Desktop/Github/Adey_FraudDetection/data/IpAddress_to_Country.csv')

    # Display columns in the datasets
    print("Columns in creditcard_df:", creditcard_df.columns)
    print("Columns in fraud_df:", fraud_df.columns)
    print("Columns in ip_df:", ip_df.columns)

    # Ensure IP columns are integers
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)

    # Merge fraud_df with ip_df based on IP address
    merged_df = pd.merge(
        fraud_df, ip_df, how='left', 
        left_on='ip_address', right_on='lower_bound_ip_address'
    )

    # Drop columns not needed after merging
    merged_df.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, inplace=True)

    # Check if 'transaction_amount' column exists
    if 'purchase_value' not in merged_df.columns:
        print("Column 'purchase_value' not found. Available columns are:", merged_df.columns)
    else:
        merged_df['transaction_amount'] = merged_df['purchase_value']
    
    # Feature engineering
    merged_df['is_high_risk'] = np.where(merged_df['transaction_amount'] > 1000, 1, 0)

    # One-hot encoding for categorical features
    merged_df = pd.get_dummies(merged_df, columns=['source', 'browser', 'sex'], drop_first=True)

    # Summary statistics
    print("Summary statistics:\n", merged_df.describe())

    # Return preprocessed data
    return merged_df

# Function for exploratory data analysis
def exploratory_data_analysis(df):
    # Plot histogram of transaction amounts if column exists
    if 'transaction_amount' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['transaction_amount'], bins=50)
        plt.title('Transaction Amount Distribution')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print("Column 'transaction_amount' not found for EDA.")

# Function to train and evaluate model with cross-validation
def model_training_and_evaluation(df):
    # Selecting features and target variable
    X = df.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'purchase_value'])
    y = df['class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    # Print cross-validation results
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", cv_scores.mean())

    # Fit the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Print classification report and confusion matrix
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Main execution
if __name__ == '__main__':
    # Run the data preprocessing pipeline
    merged_df = preprocess_data()

    # Conduct EDA
    exploratory_data_analysis(merged_df)

    # Train and evaluate model
    model_training_and_evaluation(merged_df)

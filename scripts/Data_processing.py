# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

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

    # Data Cleaning: Handle missing values
    fraud_df.dropna(inplace=True)  # Drop rows with missing values
    ip_df.dropna(inplace=True)
    
    # Remove duplicates
    fraud_df.drop_duplicates(inplace=True)
    ip_df.drop_duplicates(inplace=True)

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
    if 'purchase_value' in merged_df.columns:
        merged_df['transaction_amount'] = merged_df['purchase_value']
    
    # Feature engineering: Time-based features
    merged_df['purchase_time'] = pd.to_datetime(merged_df['purchase_time'])
    merged_df['hour_of_day'] = merged_df['purchase_time'].dt.hour
    merged_df['day_of_week'] = merged_df['purchase_time'].dt.dayofweek

    # Feature engineering: Transaction frequency
    merged_df['transaction_count'] = merged_df.groupby('user_id')['transaction_amount'].transform('count')

    # One-hot encoding for categorical features
    merged_df = pd.get_dummies(merged_df, columns=['source', 'browser', 'sex'], drop_first=True)

    # Normalization and scaling
    scaler = StandardScaler()
    if 'transaction_amount' in merged_df.columns:
        merged_df['transaction_amount'] = scaler.fit_transform(merged_df[['transaction_amount']])

    # Summary statistics
    print("Summary statistics:\n", merged_df.describe())

    # Return preprocessed data
    return merged_df

# Function for exploratory data analysis
# Updated exploratory_data_analysis function
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

    # Select only numeric columns for the correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Check if there are any numeric columns available
    if not numeric_df.empty:
        # Bivariate analysis: Correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
    else:
        print("No numeric columns available for the correlation heatmap.")

if __name__ == '__main__':
    # Run the data preprocessing pipeline
    merged_df = preprocess_data()

    # Conduct EDA
    exploratory_data_analysis(merged_df)

   

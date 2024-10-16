import pandas as pd
from datetime import datetime

# File path for the dataset
file_path = r'C:/Users/user/Desktop/Github/Adey_FraudDetection/data/creditcard.csv'

def load_data(path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(path)

def handle_missing_values(df):
    """
    Handle missing values by dropping rows with missing values.
    """
    df = df.dropna()
    return df

def convert_data_types(df):
    """
    Convert columns to appropriate data types.
    """
    # Convert the 'Time' column to datetime format based on the time elapsed
    df['Time'] = pd.to_datetime(df['Time'], unit='s', origin='unix')
    return df

def data_cleaning(df):
    """
    Clean the data by removing duplicates and correcting data types.
    """
    # Remove duplicate rows
    df = df.drop_duplicates()
    # Convert data types
    df = convert_data_types(df)
    return df

def feature_engineering(df):
    """
    Create new features for the dataset.
    """
    # Create hour_of_day and day_of_week features based on the 'Time' column
    df['hour_of_day'] = df['Time'].dt.hour
    df['day_of_week'] = df['Time'].dt.dayofweek
    return df

def preprocess_data(path):
    """
    Main function to preprocess the data.
    """
    df = load_data(path)
    df = handle_missing_values(df)
    df = data_cleaning(df)
    df = feature_engineering(df)
    return df

if __name__ == "__main__":
    # Run preprocessing and save the preprocessed data
    preprocessed_df = preprocess_data(file_path)
    preprocessed_df.to_csv(r'C:/Users/user/Desktop/Github/Adey_FraudDetection/data/creditcard.csv', index=False)
    print("Data preprocessing complete. The preprocessed file has been saved.")

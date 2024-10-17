import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def prepare_data():
    # Load datasets
    creditcard_df = pd.read_csv(r'C:/Users/user/Desktop/Github/Adey_FraudDetection/data/creditcard.csv')
    fraud_df = pd.read_csv(r'C:/Users/user/Desktop/Github/Adey_FraudDetection/data/Fraud_Data.csv')

    # Handle missing values, duplicates, and data types
    # (Implement your data cleaning steps here)

    # Drop non-numeric columns (or handle them accordingly)
    creditcard_df = creditcard_df.select_dtypes(include=[np.number])  # Keep only numeric columns
    fraud_df = fraud_df.select_dtypes(include=[np.number])  # Keep only numeric columns

    # Feature and target separation
    X_cc = creditcard_df.drop('Class', axis=1)
    y_cc = creditcard_df['Class']

    X_fraud = fraud_df.drop('class', axis=1)
    y_fraud = fraud_df['class']

    # Train-Test Split
    X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(X_cc, y_cc, test_size=0.2, random_state=42)
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_cc = scaler.fit_transform(X_train_cc)
    X_test_cc = scaler.transform(X_test_cc)
    X_train_fraud = scaler.fit_transform(X_train_fraud)
    X_test_fraud = scaler.transform(X_test_fraud)

    return (X_train_cc, X_test_cc, y_train_cc, y_test_cc), (X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud)


def define_and_train_models(X_train_cc, y_train_cc, X_test_cc, y_test_cc, 
                            X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud):
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),  # Increased max_iter
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
    }

    results = {}

    # Train and evaluate models
    for model_name, model in models.items():
        if model is not None:  # Ensure the model is not None
            print(f"Training {model_name}...")

            # Train on Credit Card Data
            model.fit(X_train_cc, y_train_cc)
            # Predict and evaluate Credit Card Data
            y_pred_cc = model.predict(X_test_cc)
            accuracy_cc = accuracy_score(y_test_cc, y_pred_cc)

            # Store results in a dictionary
            results[model_name] = {'Credit Card Accuracy': accuracy_cc}

            # Train on Fraud Data
            model.fit(X_train_fraud, y_train_fraud)
            # Predict and evaluate Fraud Data
            y_pred_fraud = model.predict(X_test_fraud)
            accuracy_fraud = accuracy_score(y_test_fraud, y_pred_fraud)

            # Add fraud accuracy to the results
            results[model_name]['Fraud Accuracy'] = accuracy_fraud

            print(f"Accuracy (Credit Card Data): {accuracy_cc}")
            print(f"Accuracy (Fraud Data): {accuracy_fraud}\n")

    return results

if __name__ == "__main__":
    # Prepare data
    (X_train_cc, X_test_cc, y_train_cc, y_test_cc), (X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud) = prepare_data()

    # Train models and capture results
    results = define_and_train_models(X_train_cc, y_train_cc, X_test_cc, y_test_cc,
                                      X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud)

    # Log results with MLflow
    mlflow.set_experiment("Fraud Detection Models")
    with mlflow.start_run():
        for model_name, metrics in results.items():
            mlflow.log_metric(f"{model_name} Accuracy (Credit Card)", metrics['Credit Card Accuracy'])
            mlflow.log_metric(f"{model_name} Accuracy (Fraud)", metrics['Fraud Accuracy'])
    
    # Visualization of Model Accuracies
    model_names = list(results.keys())
    credit_card_accuracies = [metrics['Credit Card Accuracy'] for metrics in results.values()]
    fraud_accuracies = [metrics['Fraud Accuracy'] for metrics in results.values()]

    # Create a DataFrame for better visualization
    accuracy_df = pd.DataFrame({
        'Model': model_names,
        'Credit Card Accuracy': credit_card_accuracies,
        'Fraud Accuracy': fraud_accuracies
    })

    print(accuracy_df)

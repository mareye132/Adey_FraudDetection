import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, SimpleRNN, Input

def prepare_data():
    # Load datasets
    creditcard_df = pd.read_csv(r'C:/Users/user/Desktop/Github/Adey_FraudDetection/data/creditcard.csv')
    fraud_df = pd.read_csv(r'C:/Users/user/Desktop/Github/Adey_FraudDetection/data/Fraud_Data.csv')

    # Handle missing values, duplicates, and data types
    # (Implement your data cleaning steps here, if necessary)

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

def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_rnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def define_and_train_models(X_train_cc, y_train_cc, X_test_cc, y_test_cc, 
                            X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud):
    
    # Define traditional ML models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Multi-Layer Perceptron': MLPClassifier(max_iter=2000)
    }

    results = {}

    # Train and evaluate traditional models
    for model_name, model in models.items():
        print(f"Training {model_name}...")

        # Train on Credit Card Data
        model.fit(X_train_cc, y_train_cc)
        y_pred_cc = model.predict(X_test_cc)
        accuracy_cc = accuracy_score(y_test_cc, y_pred_cc)

        # Store results
        results[model_name] = {'Credit Card Accuracy': accuracy_cc}

        # Train on Fraud Data
        model.fit(X_train_fraud, y_train_fraud)
        y_pred_fraud = model.predict(X_test_fraud)
        accuracy_fraud = accuracy_score(y_test_fraud, y_pred_fraud)

        results[model_name]['Fraud Accuracy'] = accuracy_fraud

        print(f"Accuracy (Credit Card Data): {accuracy_cc}")
        print(f"Accuracy (Fraud Data): {accuracy_fraud}\n")

    # Define and train deep learning models
    deep_learning_models = {
        'CNN': build_cnn((X_train_cc.shape[1], 1)),
        'RNN': build_rnn((X_train_cc.shape[1], 1)),
        'LSTM': build_lstm((X_train_cc.shape[1], 1))
    }

    # Reshape data for deep learning models (CNN, RNN, LSTM expect 3D input)
    X_train_cc_reshaped = X_train_cc.reshape(X_train_cc.shape[0], X_train_cc.shape[1], 1)
    X_test_cc_reshaped = X_test_cc.reshape(X_test_cc.shape[0], X_test_cc.shape[1], 1)
    X_train_fraud_reshaped = X_train_fraud.reshape(X_train_fraud.shape[0], X_train_fraud.shape[1], 1)
    X_test_fraud_reshaped = X_test_fraud.reshape(X_test_fraud.shape[0], X_test_fraud.shape[1], 1)

    for model_name, model in deep_learning_models.items():
        print(f"Training {model_name}...")

        # Train on Credit Card Data
        model.fit(X_train_cc_reshaped, y_train_cc, epochs=10, batch_size=32, verbose=0)
        _, accuracy_cc = model.evaluate(X_test_cc_reshaped, y_test_cc, verbose=0)

        # Train on Fraud Data
        model.fit(X_train_fraud_reshaped, y_train_fraud, epochs=10, batch_size=32, verbose=0)
        _, accuracy_fraud = model.evaluate(X_test_fraud_reshaped, y_test_fraud, verbose=0)

        results[model_name] = {'Credit Card Accuracy': accuracy_cc, 'Fraud Accuracy': accuracy_fraud}

        print(f"Accuracy (Credit Card Data): {accuracy_cc}")
        print(f"Accuracy (Fraud Data): {accuracy_fraud}\n")

    return results

if __name__ == "__main__":
    # Prepare data
    (X_train_cc, X_test_cc, y_train_cc, y_test_cc), (X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud) = prepare_data()

    # Define and train models
    results = define_and_train_models(X_train_cc, y_train_cc, X_test_cc, y_test_cc,
                                      X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud)

    # Log results with MLflow
    mlflow.set_experiment("Fraud Detection Models")
    with mlflow.start_run():
        for model_name, metrics in results.items():
            mlflow.log_metric(f"{model_name} Accuracy (Credit Card)", metrics['Credit Card Accuracy'])
            mlflow.log_metric(f"{model_name} Accuracy (Fraud)", metrics['Fraud Accuracy'])

    # Create a DataFrame for the results
    accuracy_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Credit Card Accuracy': [metrics['Credit Card Accuracy'] for metrics in results.values()],
        'Fraud Accuracy': [metrics['Fraud Accuracy'] for metrics in results.values()]
    })

    print(accuracy_df)

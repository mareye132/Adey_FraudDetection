import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

def prepare_data():
    # Load datasets
    creditcard_df = pd.read_csv(r'/content/drive/MyDrive/Adey_FraudDetection/data/creditcard.csv')
    fraud_df = pd.read_csv(r'/content/drive/MyDrive/Adey_FraudDetection/data/Fraud_Data.csv')

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

    # Scale the features (fit on training data only)
    scaler = StandardScaler()
    X_train_cc = scaler.fit_transform(X_train_cc)
    X_test_cc = scaler.transform(X_test_cc)
    X_train_fraud = scaler.fit_transform(X_train_fraud)
    X_test_fraud = scaler.transform(X_test_fraud)

    return (X_train_cc, X_test_cc, y_train_cc, y_test_cc), (X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud)

def build_and_train_cnn(X_train, y_train, X_test):
    # Reshape input for CNN
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build CNN model
    model = Sequential()
    model.add(Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Predict and return rounded values for accuracy calculation
    return np.round(model.predict(X_test_reshaped)).flatten()

def build_and_train_rnn(X_train, y_train, X_test):
    # Reshape input for RNN
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build RNN model
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train_reshaped.shape[1], 1), return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Predict and return rounded values for accuracy calculation
    return np.round(model.predict(X_test_reshaped)).flatten()

def build_and_train_lstm(X_train, y_train, X_test):
    # Reshape input for LSTM
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train_reshaped.shape[1], 1), return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Predict and return rounded values for accuracy calculation
    return np.round(model.predict(X_test_reshaped)).flatten()

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

        # Train on Fraud Data
        model.fit(X_train_fraud, y_train_fraud)
        y_pred_fraud = model.predict(X_test_fraud)
        accuracy_fraud = accuracy_score(y_test_fraud, y_pred_fraud)

        # Store results
        results[model_name] = {'Credit Card Accuracy': accuracy_cc, 'Fraud Accuracy': accuracy_fraud}

        print(f"Accuracy (Credit Card Data): {accuracy_cc}")
        print(f"Accuracy (Fraud Data): {accuracy_fraud}\n")

        # Log model and metrics with MLflow
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("Model Type", model_name)
            mlflow.log_metric("Credit Card Accuracy", accuracy_cc)
            mlflow.log_metric("Fraud Accuracy", accuracy_fraud)
            mlflow.sklearn.log_model(model, model_name)

    # Build and evaluate CNN
    print("Training CNN...")
    cnn_predictions = build_and_train_cnn(X_train_fraud, y_train_fraud, X_test_fraud)
    cnn_accuracy = accuracy_score(y_test_fraud, cnn_predictions)
    results['CNN'] = {'Credit Card Accuracy': None, 'Fraud Accuracy': cnn_accuracy}
    
    with mlflow.start_run(run_name='CNN'):
        mlflow.log_param("Model Type", "CNN")
        mlflow.log_metric("Fraud Accuracy", cnn_accuracy)
    
    # Build and evaluate RNN
    print("Training RNN...")
    rnn_predictions = build_and_train_rnn(X_train_fraud, y_train_fraud, X_test_fraud)
    rnn_accuracy = accuracy_score(y_test_fraud, rnn_predictions)
    results['RNN'] = {'Credit Card Accuracy': None, 'Fraud Accuracy': rnn_accuracy}
    
    with mlflow.start_run(run_name='RNN'):
        mlflow.log_param("Model Type", "RNN")
        mlflow.log_metric("Fraud Accuracy", rnn_accuracy)

    # Build and evaluate LSTM
    print("Training LSTM...")
    lstm_predictions = build_and_train_lstm(X_train_fraud, y_train_fraud, X_test_fraud)
    lstm_accuracy = accuracy_score(y_test_fraud, lstm_predictions)
    results['LSTM'] = {'Credit Card Accuracy': None, 'Fraud Accuracy': lstm_accuracy}
    
    with mlflow.start_run(run_name='LSTM'):
        mlflow.log_param("Model Type", "LSTM")
        mlflow.log_metric("Fraud Accuracy", lstm_accuracy)

    return results

if __name__ == "__main__":
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri('/content/drive/MyDrive/Adey_FraudDetection/notebooks/mlruns')  # Update path here
    mlflow.set_experiment("Fraud Detection Models")

    # Prepare data
    (X_train_cc, X_test_cc, y_train_cc, y_test_cc), (X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud) = prepare_data()

    # Define and train models
    results = define_and_train_models(X_train_cc, y_train_cc, X_test_cc, y_test_cc,
                                      X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud)

    print("\nModel Performance Summary:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

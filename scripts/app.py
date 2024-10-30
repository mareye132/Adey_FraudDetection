# app.py
from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load data from CSV
def load_data():
    df = pd.read_csv('C:/Users/user/Desktop/Github/Adey_FraudDetection/data/Fraud_Data.csv')
    return df

# Endpoint for summary statistics
@app.route('/api/summary', methods=['GET'])
def summary():
    df = load_data()
    total_transactions = len(df)
    total_fraud = df[df['is_fraud'] == 1].shape[0]
    fraud_percentage = (total_fraud / total_transactions) * 100
    
    summary_data = {
        "total_transactions": total_transactions,
        "total_fraud_cases": total_fraud,
        "fraud_percentage": fraud_percentage
    }
    return jsonify(summary_data)

# Endpoint for fraud trends over time
@app.route('/api/trends', methods=['GET'])
def fraud_trends():
    df = load_data()
    fraud_over_time = df[df['is_fraud'] == 1].groupby('date')['is_fraud'].count()
    trends = fraud_over_time.to_dict()
    return jsonify(trends)

# Additional endpoints for device/browser analysis, geography, etc. can be added similarly

if __name__ == '__main__':
    app.run(debug=True)

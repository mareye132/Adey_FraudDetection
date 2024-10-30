# serve_model.py

import pickle
from flask import Flask, request, jsonify
import logging

# Initialize Flask application
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model from the specified path
model_path = "C:/Users/user/Desktop/Github/Adey_FraudDetection/mlruns/937789298497431940/6f3ae6530918427ca6449df2ed22ea83/artifacts/Random Forest/model.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Welcome to the Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json(force=True)
    
    # Extract features from the request
    features = data['features']
    
    # Make a prediction
    prediction = model.predict([features]).tolist()

    # Log the incoming request and prediction
    logging.info(f'Incoming request: {data}')
    logging.info(f'Prediction: {prediction}')
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

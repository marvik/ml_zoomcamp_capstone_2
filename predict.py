import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

# Load the model and preprocessor
MODEL_FILE = 'optimized_model.h5'  
PREPROCESSOR_FILE = 'preprocessor.pkl'  

# Define custom object for loading the model
custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError()  
}

# Load the model with custom objects
model = load_model(MODEL_FILE, custom_objects=custom_objects)
preprocessor = joblib.load(PREPROCESSOR_FILE)

# Features used during training 
numerical_features = ['latitude', 'longitude', 'minimum_nights',
                      'number_of_reviews', 'reviews_per_month',
                      'calculated_host_listings_count', 'availability_365']
categorical_features = ['neighbourhood_group', 'room_type', 'room_neighborhood']
features = numerical_features + categorical_features

app = Flask('airbnb_price_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    listing_data = request.get_json()

    # Convert to DataFrame
    try:
        df = pd.DataFrame([listing_data], columns=features)
    except ValueError:
        return jsonify({'error': 'Missing or mismatched features in input data'}), 400

    # Preprocess the data
    X = preprocessor.transform(df)

    # Make prediction
    prediction = model.predict(X).flatten()[0]
    price = np.expm1(prediction)  # Inverse transform

    result = {
        'predicted_price': float(price)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
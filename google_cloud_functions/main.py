import os
from flask import jsonify, request
import numpy as np
from utils import run_svm_prediction, base64_to_object


def predict(request):
    # Get the feature matrix from the request body
    # Format of body must be { "features": [0.1, 0.2, 0.3, 0.4] }

    try:
        data = request.get_json()
        X = np.array(data["features"]).reshape(1, -1)
    except Exception as e:
        print(f"Error parsing request body: {e}")
        return jsonify({"error": "Invalid request body"}), 400

    # Get the model, scaler, and selected_features from environment variables
    loaded_svm_model = base64_to_object(os.environ.get('MODEL_BASE64'))
    scaler = base64_to_object(os.environ.get('SCALER_BASE64'))
    selected_features = base64_to_object(os.environ.get('SELECTED_FEATURES_BASE64'))

    # Create a new feature matrix with only the selected features
    X_selected = X[:, selected_features]

    # Scale the feature matrix to have mean=0 and variance=1
    X_scaled = scaler.transform(X_selected)

    # Use the loaded model for predictions
    y_pred = run_svm_prediction(loaded_svm_model, X_scaled)

    # Return a response indicating whether the model predicts that it is a positive case
    if y_pred == 1:
        result = {"result": "ChatGPT"}
    else:
        result = {"result": "Human"}

    return jsonify(result), 200

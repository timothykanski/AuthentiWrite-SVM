# lambda.py

import json
import numpy as np
from constants import MODEL_BASE64, SCALER_BASE64, SELECTED_FEATURES_BASE64
from utils import run_svm_prediction, base64_to_object


def lambda_predictor(event, context):
    # Get the feature matrix from the request body
    try:
        data = json.loads(event["body"])
        X = np.array(data["features"]).reshape(1, -1)
    except Exception as e:
        print(f"Error parsing request body: {e}")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid request body"}),
        }

    # Get the model, scaler, and selected_features from constants.py
    loaded_svm_model = base64_to_object(MODEL_BASE64)
    scaler = base64_to_object(SCALER_BASE64)
    selected_features = base64_to_object(SELECTED_FEATURES_BASE64)

    # Create a new feature matrix with only the selected features
    X_selected = X[:, selected_features]

    # Scale the feature matrix to have mean=0 and variance=1
    X_scaled = scaler.transform(X_selected)

    # Use the loaded model for predictions
    y_pred = run_svm_prediction(loaded_svm_model, X_scaled)

    # Return a response indicating whether the model predicts that it is a positive case
    if y_pred == 1:
        result = {"prediction": "ChatGPT"}
    else:
        result = {"prediction": "Human"}

    return {"statusCode": 200, "body": json.dumps(result)}

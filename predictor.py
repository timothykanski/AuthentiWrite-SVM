# predictor.py

from constants import MODEL_BASE64, SCALER_BASE64, SELECTED_FEATURES_BASE64
from evaluation_metrics import EvaluationMetrics
from utils import load_mldataset, run_svm_prediction, print_average_metrics, base64_to_object
from memory_profiler import profile


@profile
def main():
    # Load a test dataset
    X, y = load_mldataset("C:\\AuthentiWrite\\datasets\\example_3_dataset.json")
    print("New Set")
    print(f"Number of Features: {X.shape[1]}")
    print(f"Number of Samples: {len(X)}")

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

    # Calculate the evaluation metrics
    evaluation_metrics_list = []
    seed = 0
    c = 0
    evaluation_metrics = EvaluationMetrics(y, y_pred, 0, 0)

    # Append to the list (useful when running this in a loop)
    evaluation_metrics_list.append(evaluation_metrics)

    # Get other metrics
    print_average_metrics(evaluation_metrics_list)


if __name__ == "__main__":
    main()

# Import necessary libraries
import pickle
from memory_profiler import profile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from evaluation_metrics import EvaluationMetrics
from utils import load_mldataset, select_features_kbest, print_average_metrics, calculate_max_precision, \
    run_svm_prediction


@profile
def main():
    X, y = load_mldataset("C:\\AuthentiWrite\\datasets\\example_2_dataset.json")
    print("TrainTest Set")
    print(f"Number of Features: {X.shape[1]}")
    print(f"Number of Samples: {len(X)}")

    # Create a new feature matrix with only the selected features
    selected_features = select_features_kbest(X, y, 10)
    X_selected = X[:, selected_features]

    # Save the selected features mask
    with open('selected_features.pkl', 'wb') as sf_file:
        pickle.dump(selected_features, sf_file)

    evaluation_metrics_list = []
    seed = 2
    # c = 10 ** (j - 1)
    c = 0.1

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=seed)

    # Scale the feature matrix to have mean=0 and variance=1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Save the StandardScaler
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    # Scale the test dataset
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM model
    svm_model = SVC(kernel='linear', C=c)
    svm_model.fit(X_train_scaled, y_train)

    # Save the trained model to a file
    with open('svm_model.pkl', 'wb') as model_file:
        pickle.dump(svm_model, model_file)

    # Perform inference using the trained model
    y_pred = run_svm_prediction(svm_model, X_test_scaled)

    # Calculate the evaluation metrics
    evaluation_metrics = EvaluationMetrics(y_test, y_pred, c, seed)

    # Append to the list (useful when running this in a loop)
    evaluation_metrics_list.append(evaluation_metrics)

    # Get other metrics
    print_average_metrics(evaluation_metrics_list)


if __name__ == "__main__":
    main()

# utils.py
import base64
import io
import json
import pickle

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def load_mldataset(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    feature_matrix = np.array(data['Features'])  # convert list to numpy array
    target_vector = np.array(data['Targets'])  # convert list to numpy array

    return feature_matrix, target_vector


def select_features_kbest(X, y, k):
    # Define the mutual information-based feature selector
    selector = SelectKBest(mutual_info_classif, k=k)

    # Fit the selector
    selector.fit(X, y)

    # Get the selected features
    selected_features = selector.get_support()

    return selected_features


def print_average_metrics(evaluation_metrics_list):
    accuracy_result = [evaluation_metrics.accuracy for evaluation_metrics in evaluation_metrics_list]
    precision_result = [evaluation_metrics.precision for evaluation_metrics in evaluation_metrics_list]
    recall_result = [evaluation_metrics.recall for evaluation_metrics in evaluation_metrics_list]
    f1_result = [evaluation_metrics.f1 for evaluation_metrics in evaluation_metrics_list]
    cm_result = [evaluation_metrics.cm for evaluation_metrics in evaluation_metrics_list]

    # Calculate Max Precision
    max_precision = calculate_max_precision(evaluation_metrics_list)
    print(f"Max Precision: {max_precision[0]}, C:{max_precision[1]}, seed:{max_precision[2]}")

    avg_accuracy = sum(accuracy_result) / len(accuracy_result)
    avg_precision = sum(precision_result) / len(precision_result)
    avg_recall = sum(recall_result) / len(recall_result)
    avg_f1 = sum(f1_result) / len(f1_result)
    avg_cm = sum(cm_result) / len(cm_result)

    # Print the results
    print(f"Average accuracy: {avg_accuracy:.2f}")
    print(f"Average precision: {avg_precision:.2f}")
    print(f"Average recall: {avg_recall:.2f}")
    print(f"Average F1-score: {avg_f1:.2f}")
    print(f"Average confusion matrix:\n{avg_cm}")


def calculate_max_precision(evaluation_metrics_list):
    max_p = 0
    max_precision_c = 0
    max_precision_seed = 0
    for m in evaluation_metrics_list:
        if m.precision > max_p:
            max_p = m.precision
            max_precision_c = m.C
            max_precision_seed = m.seed
    return max_p, max_precision_c, max_precision_seed


def run_svm_prediction(svm_model, X_test_scaled):
    return svm_model.predict(X_test_scaled)


def file_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def base64_to_object(base64_str):
    return pickle.load(io.BytesIO(base64.b64decode(base64_str)))

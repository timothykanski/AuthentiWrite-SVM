# Import necessary libraries
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# from sklearn import datasets
# Load a sample dataset (Iris dataset)
# wine = datasets.load_wine()
# X = wine.data
# y = wine.target
def load_mldataset(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    feature_matrix = np.array(data['Features'])  # convert list to numpy array
    target_vector = np.array(data['Targets'])  # convert list to numpy array

    return feature_matrix, target_vector


class EvaluationMetrics:
    def __init__(self, y_actual, y_predicted, c, seed):
        self.seed = i
        self.C = c
        self.accuracy = accuracy_score(y_actual, y_predicted)
        self.precision = precision_score(y_actual, y_predicted)
        self.recall = recall_score(y_actual, y_predicted)
        self.f1 = f1_score(y_actual, y_predicted)
        self.cm = confusion_matrix(y_actual, y_predicted)

    def get_properties(self):
        return self.accuracy, self.precision, self.recall, self.f1, self.cm, self.C, self.seed


def calculate_average_metrics(evaluation_metrics_list):
    accuracy_result = [evaluation_metrics.accuracy for evaluation_metrics in evaluation_metrics_list]
    precision_result = [evaluation_metrics.precision for evaluation_metrics in evaluation_metrics_list]
    recall_result = [evaluation_metrics.recall for evaluation_metrics in evaluation_metrics_list]
    f1_result = [evaluation_metrics.f1 for evaluation_metrics in evaluation_metrics_list]
    cm_result = [evaluation_metrics.cm for evaluation_metrics in evaluation_metrics_list]

    avg_accuracy = sum(accuracy_result) / len(accuracy_result)
    avg_precision = sum(precision_result) / len(precision_result)
    avg_recall = sum(recall_result) / len(recall_result)
    avg_f1 = sum(f1_result) / len(f1_result)
    avg_cm = sum(cm_result) / len(cm_result)

    return avg_accuracy, avg_precision, avg_recall, avg_f1, avg_cm


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


X, y = load_mldataset("C:\\AuthentiWrite\\datasets\\example_2_dataset.json")
print("TrainTest Set")
print(f"Number of Features: {X.shape[1]}")
print(f"Number of Samples: {len(X)}")

# Define the mutual information-based feature selector
selector = SelectKBest(mutual_info_classif, k=10)

# Fit the selector
selector.fit(X, y)

# Get the selected features
selected_features = selector.get_support()

# Create a new feature matrix with only the selected features
X_selected = X[:, selected_features]

X_new, y_new = load_mldataset("C:\\AuthentiWrite\\datasets\\example_3_dataset.json")
print("New Set")
print(f"Number of Features: {X_new.shape[1]}")
print(f"Number of Samples: {len(X_new)}")

# Create a new feature matrix with only the selected features
X_new_selected = X_new[:, selected_features]

accuracy_result = []
precision_result = []
recall_result = []
f1_result = []
cm_result = []
evaluation_metric_list = []


for j in range(10):  # 10

    for i in range(10):  # 10
        k = 10 ** (j - 1)
        # k = 0.1

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=2)

        # Test new set
        X_test = X_new_selected
        y_test = y_new

        # Scale the feature matrix to have mean=0 and variance=1
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the SVM model
        svm_model = SVC(kernel='linear', C=0.1)
        svm_model.fit(X_train_scaled, y_train)

        # Perform inference using the trained model
        y_pred = svm_model.predict(X_test_scaled)

        # Calculate the evaluation metrics
        evaluation_metrics = EvaluationMetrics(y_test, y_pred, k, i)
        evaluation_metric_list.append(evaluation_metrics)
        # properties = evaluation_metrics.get_properties()
        # property_list.append(properties)

        # accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        # cm = confusion_matrix(y_test, y_pred)

        # Append the results to the lists
        # accuracy_result.append(accuracy)
        # precision_result.append(precision)
        # recall_result.append(recall)
        # f1_result.append(f1)
        # cm_result.append(cm)

# Calculate the average of the evaluation metrics
# avg_accuracy = sum(accuracy_result) / len(accuracy_result)
# avg_precision = sum(precision_result) / len(precision_result)
# avg_recall = sum(recall_result) / len(recall_result)
# avg_f1 = sum(f1_result) / len(f1_result)
# avg_cm = sum(cm_result) / len(cm_result)

# Calculate Max Precision
max_precision = calculate_max_precision(evaluation_metric_list)
print(f"Max Precision: {max_precision[0]}, C:{max_precision[1]}, seed:{max_precision[2]}")

# Get other metrics
avg_metrics = calculate_average_metrics(evaluation_metric_list)

# Print the results
print(f"C is {k}")
print(f"random_state is {i}")
print(f"Average accuracy: {avg_metrics[0]:.2f}")
print(f"Average precision: {avg_metrics[1]:.2f}")
print(f"Average recall: {avg_metrics[2]:.2f}")
print(f"Average F1-score: {avg_metrics[3]:.2f}")
print(f"Average confusion matrix:\n{avg_metrics[4]}")



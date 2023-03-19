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

# Transform X_new to have only the selected features
X_new_selected = selector.transform(X_new)

# for j in range(6):
accuracy_result = []
precision_result = []
recall_result = []
f1_result = []
cm_result = []

#for i in range(30):
# k = 10 ** (j - 3)
k = 0.1
i = 7

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=i)

# Scale the feature matrix to have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_new_scaled = scaler.transform(X_new_selected)

# Train the SVM model
svm_model = SVC(kernel='linear', C=k)
svm_model.fit(X_train_scaled, y_train)

# Perform inference using the trained model
y_pred = svm_model.predict(X_test_scaled)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Append the results to the lists
accuracy_result.append(accuracy)
precision_result.append(precision)
recall_result.append(recall)
f1_result.append(f1)
cm_result.append(cm)

# Calculate the average of the evaluation metrics
avg_accuracy = sum(accuracy_result) / len(accuracy_result)
avg_precision = sum(precision_result) / len(precision_result)
avg_recall = sum(recall_result) / len(recall_result)
avg_f1 = sum(f1_result) / len(f1_result)
avg_cm = sum(cm_result) / len(cm_result)

# Print the results
print(f"C is {k}")
print(f"random_state is {i}")
print(f"accuracy: {accuracy:.2f}")
print(f"precision: {precision:.2f}")
print(f"recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"confusion matrix:\n{cm}")

print(f"Average accuracy: {avg_accuracy:.2f}")
print(f"Average precision: {avg_precision:.2f}")
print(f"Average recall: {avg_recall:.2f}")
print(f"Average F1-score: {avg_f1:.2f}")
print(f"Average confusion matrix:\n{avg_cm}")


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import json
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import joblib


def load_mldataset(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    feature_matrix = np.array(data['Features'])  # convert list to numpy array
    target_vector = np.array(data['Targets'])  # convert list to numpy array

    return feature_matrix, target_vector


X, y = load_mldataset("C:\\AuthentiWrite\\datasets\\example_2_dataset.json")

print(f"Number of Features: {X.shape[1]}")
print(f"Number of Samples: {len(X)}")

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

# Scale the feature matrix to have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear', C=100)
svm_model.fit(X_train_scaled, y_train)

# Save the model
filename = "C:\\AuthentiWrite\\models_SVM\\svm1"
joblib.dump(svm_model, filename)

# Load the new test datasets
X, y = load_mldataset("C:\\AuthentiWrite\\datasets\\example_3_dataset.json")

loaded_model = joblib.load(filename)

# Use the loaded model to make predictions on new data
y_pred = loaded_model.predict(X)

print(y_pred)

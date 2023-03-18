# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load a sample dataset (Iris dataset)
iris = datasets.load_iris()
X = iris.data
y = iris.target

for j in range(7):

    result_set = []

    for i in range(60):

        k = 10**(j-1)

        # print("                                          ")
        # print(f"i is {i} and j is {j} and C is {k}")

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

        # Scale the feature matrix to have mean=0 and variance=1
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the SVM model
        svm_model = SVC(kernel='linear', C=k)
        svm_model.fit(X_train_scaled, y_train)

        # Perform inference using the trained model
        y_pred = svm_model.predict(X_test_scaled)

        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # print(f'Test accuracy: {accuracy:.2f}')

        # Perform inference on a single sample
        sample = X_test_scaled[0]  # Take the first test sample
        sample_prediction = svm_model.predict([sample])
        # print(f'Sample true class: {y_test[0]}, predicted class: {sample_prediction[0]}')

        result_set.append(accuracy)

    avg = sum(result_set) / len(result_set)

    print(f"C is {k}, the average accuracy is {avg}")
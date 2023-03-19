# LearnSkLearn

SVM Machine learning for Authorship Attribution (specifcally for ChatGPT vs human)

## Setup

TODO: Describe how to set up the project and its dependencies.

## Usage

### Train the SVM model

1. Edit SVM.py to specify the training/testing dataset path and other hyperparameters, if needed.
2. Run `python SVM.py`. The script will split the dataset, train the model, and save it along with the scaler and selected features as pkl files.
3. Run `python print_model_base64.py`. The script will print the base64 encoded model, scaler, and selected features to the console.
4. Copy the output from the console and paste it into constants.py. Replace the existing base64 strings.

### Predict with the SVM model

1. Edit predictor.py to specify the input dataset path and other parameters, if needed.
2. Run `python predictor.py`. The script will load the model, scaler, and selected features from constants.py, perform predictions on the input dataset, and print the evaluation metrics to the console.

## Files

- SVM.py: Trains an SVM model using the input dataset and saves it along with the scaler and selected features as pkl files.
- print_model_base64.py: Prints the base64 encoded SVM model, scaler, and selected features to the console.
- predictor.py: Performs predictions using the SVM model loaded from constants.py.
- evaluation_metrics.py: Contains the EvaluationMetrics class to compute evaluation metrics for classification problems.
- constants.py: Stores the base64 encoded SVM model, scaler, and selected features.

## References

List any references used in the project, such as data sources, libraries, or research papers.

import base64

from utils import file_to_base64

model_base64 = file_to_base64("svm_model.pkl")
scaler_base64 = file_to_base64("scaler.pkl")
selected_features_base64 = file_to_base64("selected_features.pkl")

print("Model base64:", model_base64)
print("Scaler base64:", scaler_base64)
print("Selected Features base64:", selected_features_base64)

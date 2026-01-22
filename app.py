
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model & scaler
model = load_model("Model/Churn_Modelling.h5")
scaler = joblib.load("Model/scaler.pkl")

def predict_churn(features):
    scaled = scaler.transform([features])
    pred = model.predict(scaled)
    return 1 if pred[0][0] > 0.5 else 0

# Example input
sample = [600, 1, 1, 40, 3, 60000, 2, 1, 1, 50000]  # Replace with real values
print("Churn Prediction:", predict_churn(sample))

   
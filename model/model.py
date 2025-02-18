import joblib
import pandas as pd
import os

# Load the trained model
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "calorie_model.pkl")
    model = joblib.load(model_path)
    return model

# Predict calories burnt based on user input
def predict(input_data):
    print("Input Data for Prediction:", input_data)
    model = load_model()
    prediction = model.predict(input_data)
    print("Prediction:", prediction)
    return prediction
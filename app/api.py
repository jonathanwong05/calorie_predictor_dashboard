import pandas as pd


class CaloriePredictorAPI:
    def preprocess_input(self, gender, age, height, weight, duration, heart_rate, body_temp):
        """
        Preprocesses user inputs for model prediction.

        Args:
            gender (str): "male" or "female"
            age (int): User's age
            height (float): Height in cm
            weight (float): Weight in kg
            duration (float): Exercise duration in minutes
            heart_rate (float): Heart rate during exercise
            body_temp (float): Body temperature during exercise

        Returns:
            pd.DataFrame: Preprocessed input as a DataFrame
        """
        # Encode gender
        gender_encoded = 0 if gender.lower() == "male" else 1

        # Create a DataFrame for input
        input_data = pd.DataFrame({
            "Gender": [gender_encoded],
            "Age": [age],
            "Height": [height],
            "Weight": [weight],
            "Duration": [duration],
            "Heart_Rate": [heart_rate],
            "Body_Temp": [body_temp]
        })
        return input_data
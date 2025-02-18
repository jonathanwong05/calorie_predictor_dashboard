from model import predict
import pandas as pd

# Example input for testing
test_input = pd.DataFrame({
    "Gender": [0],  # 0 for Male, 1 for Female
    "Age": [30],
    "Height": [170],
    "Weight": [70],
    "Duration": [30],
    "Heart_Rate": [100],
    "Body_Temp": [37]
})

print("Test Input:", test_input)
print("Prediction Output:", predict(test_input))
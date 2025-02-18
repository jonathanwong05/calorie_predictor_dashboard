import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import os

# Load datasets
calories = pd.read_csv('C:/Users/jonat/Personal Projects/calorie_predictor_dashboard/data/calories.csv')
exercise = pd.read_csv('C:/Users/jonat/Personal Projects/calorie_predictor_dashboard/data/exercise.csv')

# Merge datasets
calories_data = pd.concat([exercise, calories['Calories']], axis=1)

# Encode categorical data
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# Define features (X) and target (Y)
X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
Y = calories_data['Calories']

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Evaluate the model
test_predictions = model.predict(X_test)
mae = metrics.mean_absolute_error(Y_test, test_predictions)
print("Mean Absolute Error:", mae)

# Ensure the directory exists before saving the model
model_dir = 'C:/Users/jonat/Personal Projects/calorie_predictor_dashboard/model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model
joblib.dump(model, os.path.join(model_dir, 'calorie_model.pkl'))
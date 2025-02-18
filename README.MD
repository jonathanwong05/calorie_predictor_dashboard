# Calorie Predictor Dashboard

A dynamic dashboard that predicts calorie expenditure using an XGBoost model based on 7 user inputs (Gender, Age, Height, Weight, Duration, Heart Rate, Body Temperature) and provides interactive visualizations for model explainability.

## How to Run

1. **Clone the Repository:**
   git clone https://github.com/jonathanwong05/calorie_predictor_dashboard.git
   cd calorie_predictor_dashboard

2. **Set Up a Virtual Environment and Install Dependencies:**
    pip install -r requirements.txt

3. **Train the Model (if needed):**
    python model/train_model.py

4. **Start the Dashboard**
    panel serve app/dashboard.py

5. **Open browser and go to: http://localhost:5006/dashboard**

This README provides a concise overview of the project and the steps required to run it.
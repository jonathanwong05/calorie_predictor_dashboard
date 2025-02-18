import shap
import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
import io
import base64
import sys
from pathlib import Path

# Add the root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.model import predict

pn.extension()

# SHAP Feature Impact Plot
def generate_shap_plot(model, input_data):
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

     # Generate a bar-type summary plot which shows the average absolute SHAP values
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    plt.xlabel("Average Absolute SHAP Value")
    plt.title("Feature Impact on Predicted Calories")

    # Convert plot to HTML
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return f"<img src='data:image/png;base64,{encoded}' />"

# Input vs Prediction Chart
def input_vs_prediction_chart(model, feature_name, input_data, feature_range):
    predictions = []
    for value in feature_range:
        temp_data = input_data.copy()
        temp_data[feature_name] = value
        predictions.append(predict(temp_data)[0])

    plt.figure(figsize=(8, 5))
    plt.plot(feature_range, predictions, marker='o')
    plt.title(f"{feature_name} vs. Predicted Calories")
    plt.xlabel(feature_name)
    plt.ylabel("Predicted Calories")
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"<img src='data:image/png;base64,{encoded}'/>"

# Historical Data Table
def historical_data_table(history_df):
    return pn.widgets.Tabulator(history_df, pagination='remote', page_size=10, selectable=False)
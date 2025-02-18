import panel as pn
import pandas as pd
from api import CaloriePredictorAPI
import sys
from pathlib import Path
from visualization import generate_shap_plot, input_vs_prediction_chart, historical_data_table

# Add the root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import model prediction function
from model.model import predict, load_model
model = load_model()

# Initialize Panel and API
pn.extension()
api = CaloriePredictorAPI()

# Historical Dataframe
history = pd.DataFrame(columns=["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Prediction"])

# Widgets for user inputs
gender = pn.widgets.RadioButtonGroup(name="Gender", options=["Male", "Female"], value="Male")
age = pn.widgets.IntSlider(name="Age", start=18, end=80, value=30)
height = pn.widgets.FloatSlider(name="Height (cm)", start=140, end=200, value=170)
weight = pn.widgets.FloatSlider(name="Weight (kg)", start=40, end=150, value=70)
duration = pn.widgets.FloatSlider(name="Duration (min)", start=1, end=60, value=30)
heart_rate = pn.widgets.FloatSlider(name="Heart Rate", start=60, end=180, value=100)
body_temp = pn.widgets.FloatSlider(name="Body Temperature (°C)", start=36, end=42, value=37)

# Widgets for visualization
feature_select = pn.widgets.Select(name="Select Feature", options=["Age", "Weight", "Duration", "Heart_Rate"], value="Weight")
feature_range = pn.widgets.IntRangeSlider(name="Feature Range", start=10, end=150, value=(50, 100))

# Widgets for plot customization
plot_width = pn.widgets.IntSlider(name="Plot Width", start=400, end=1500, step=100, value=800)
plot_height = pn.widgets.IntSlider(name="Plot Height", start=300, end=1200, step=100, value=600)

# Visualization Panels (initialize as empty)
shap_plot = pn.pane.HTML()
sensitivity_chart = pn.pane.HTML()
history_table = pn.widgets.Tabulator(pd.DataFrame(), selectable=False)


# Output sections
prediction_result = pn.pane.Markdown("### Predicted Calories Burnt: ")

# Callback to handle predictions
def update_prediction():
    global history
    input_data = api.preprocess_input(
        gender.value, age.value, height.value, weight.value,
        duration.value, heart_rate.value, body_temp.value
    )
    prediction = predict(input_data)

    # Log to history
    record = input_data.copy()
    record["Prediction"] = prediction[0]
    history = pd.concat([history, record], ignore_index=True)

    # Update UI elements
    prediction_result.object = f"### Predicted Calories Burnt: {prediction[0]:.2f} kcal"
    shap_plot.object = generate_shap_plot(model, input_data)
    sensitivity_chart.object = input_vs_prediction_chart(model, feature_select.value, input_data, range(*feature_range.value))
    history_table.value = historical_data_table(history)

    # Refresh data table with latest history
    history_table.value = history

# Bind widgets to callbacks
widgets = [gender, age, height, weight, duration, heart_rate, body_temp]
for widget in widgets:
    widget.param.watch(lambda event, w=widget: update_prediction(), "value")

# Dashboard layout
layout = pn.template.FastListTemplate(
    title="Calorie Predictor Dashboard",
    sidebar=[
        pn.Card(
            pn.Column(gender, age, height, weight, duration, heart_rate, body_temp),
            title="User Input",
            width=320
        ),
        pn.Card(pn.Column(plot_width, plot_height, feature_select), title="Plot Options", width=320, collapsed=True),
    ],
    main=[
        pn.Tabs(
            ("Prediction", prediction_result),
            ("SHAP Feature Impact", shap_plot),
            ("Sensitivity Analysis", sensitivity_chart),
            ("Historical Data", history_table),
            active=0
        )
    ],
    header_background="#a93226"
)

layout.servable()
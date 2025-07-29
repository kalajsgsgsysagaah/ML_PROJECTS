import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import gradio as gr

# Load and clean dataset
data = pd.read_csv("Admission_Predict.csv")
data.columns = data.columns.str.strip()

# Drop Serial No. and split features/target
X = data.drop(columns=["Serial No.", "Chance of Admit"])
y = data["Chance of Admit"]

# Define the model
model = RandomForestRegressor(random_state=42)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"Cross-Validated R² Scores: {scores}")
print(f"Mean R² Score: {scores.mean():.4f}")

# Train on full data after CV
model.fit(X, y)

# Define prediction function for Gradio
def predict_admission(gre, toefl, rating, sop, lor, cgpa, research):
    features = [[gre, toefl, rating, sop, lor, cgpa, research]]
    prediction = model.predict(features)[0]
    return f"🎯 Predicted Chance of Admit: {prediction * 100:.2f}%"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_admission,
    inputs=[
        gr.Number(label="GRE Score"),
        gr.Number(label="TOEFL Score"),
        gr.Number(label="University Rating"),
        gr.Number(label="SOP (1-5)"),
        gr.Number(label="LOR (1-5)"),
        gr.Number(label="CGPA (out of 10)"),
        gr.Radio([0, 1], label="Research Experience (0 = No, 1 = Yes)")
    ],
    outputs="text",
    title="🎓 Admission Chance Predictor",
    description="Enter your academic profile to estimate your chance of admission."
)

iface.launch()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import gradio as gr

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv("Admission_Predict_Final_With_Degree.csv")
data.columns = data.columns.str.strip()

# Remove Program_Competitiveness if exists
if "Program_Competitiveness" in data.columns:
    data = data.drop(columns=["Program_Competitiveness"])

# -----------------------------
# HANDLE OPTIONAL EXAMS
# -----------------------------
exam_columns = ["IELTS", "TOEFL", "PTE", "DET", "GRE"]
data[exam_columns] = data[exam_columns].fillna(0)

# -----------------------------
# ENCODE CATEGORICAL FEATURES
# -----------------------------
data["Country_Aiming"] = data["Country_Aiming"].astype("category")
data["Country_Encoded"] = data["Country_Aiming"].cat.codes
country_map = dict(zip(data["Country_Aiming"].cat.categories,
                       data["Country_Aiming"].cat.codes.unique()))

data["Exam_Type"] = data["Exam_Type"].astype("category")
data["Exam_Encoded"] = data["Exam_Type"].cat.codes
exam_map = dict(zip(data["Exam_Type"].cat.categories,
                    data["Exam_Type"].cat.codes.unique()))

data["Degree_Level"] = data["Degree_Level"].astype("category")
data["Degree_Encoded"] = data["Degree_Level"].cat.codes
degree_map = dict(zip(data["Degree_Level"].cat.categories,
                      data["Degree_Level"].cat.codes.unique()))

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X_cols = [
    "Degree_Encoded",
    "Work_Experience_Years",
    "CGPA",
    "SOP",
    "LOR",
    "Research",
    "IELTS",
    "TOEFL",
    "PTE",
    "DET",
    "GRE",
    "Exam_Encoded",
    "Country_Encoded"
]

X = data[X_cols]
y = data["Chance_of_Admit"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

print("Training Model...")
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print("Model R² Score:", round(scores.mean(), 4))

model.fit(X, y)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_admission(
    degree,
    exam_type,
    exam_score,
    work_exp,
    cgpa,
    sop,
    lor,
    research,
    country
):

    # -----------------------------
    # VALIDATE EXAM SCORE
    # -----------------------------
    exam_limits = {
        "IELTS": (0, 9),
        "TOEFL": (0, 120),
        "PTE": (10, 90),
        "DET": (10, 160),
        "GRE": (260, 340)
    }

    min_score, max_score = exam_limits[exam_type]

    if exam_score < min_score or exam_score > max_score:
        return f"❌ Invalid score for {exam_type}. Allowed range: {min_score} - {max_score}"

    # Reset exam scores
    ielts = toefl = pte = det = gre = 0

    if exam_type == "IELTS":
        ielts = exam_score
    elif exam_type == "TOEFL":
        toefl = exam_score
    elif exam_type == "PTE":
        pte = exam_score
    elif exam_type == "DET":
        det = exam_score
    elif exam_type == "GRE":
        gre = exam_score

    features = [[
        degree_map[degree],
        work_exp,
        cgpa,
        sop,
        lor,
        research,
        ielts,
        toefl,
        pte,
        det,
        gre,
        exam_map[exam_type],
        country_map[country]
    ]]

    prediction = model.predict(features)[0]
    prediction = max(0, min(prediction, 1))

    return f"🎯 Predicted Chance of Admit: {prediction*100:.2f}%"
# -----------------------------
# GRADIO UI
# -----------------------------
iface = gr.Interface(
    fn=predict_admission,
    inputs=[
        gr.Dropdown(list(degree_map.keys()), label="Degree Level"),
        gr.Dropdown(list(exam_map.keys()), label="Exam Type"),
        gr.Number(label="Exam Score"),
        gr.Slider(0, 10, step=1, label="Work Experience (Years)"),
        gr.Slider(6.0, 10.0, step=0.1, label="CGPA"),
        gr.Slider(1.0, 5.0, step=0.5, label="SOP Strength"),
        gr.Slider(1.0, 5.0, step=0.5, label="LOR Strength"),
        gr.Radio([0, 1], label="Research (0=No, 1=Yes)"),
        gr.Dropdown(list(country_map.keys()), label="Country Applying To")
    ],
    outputs="text",
    title="🎓 Global Admission Predictor (Simplified)",
    description="Degree Level + Experience + Academics + Optional Exams"
)

iface.launch(server_name="0.0.0.0", server_port=10000)

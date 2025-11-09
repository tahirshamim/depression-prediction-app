import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import joblib

# --- Load trained model and feature names ---
model = XGBClassifier()
model.load_model("xgb_no_prof_degree.json")

feature_names = joblib.load("xgb_no_prof_degree_features.joblib")

st.set_page_config(page_title="Depression Prediction", page_icon="🧠", layout="centered")
st.title("🧠 Mental Health Depression Prediction")
st.write("This app predicts the likelihood of depression based on academic, work, and lifestyle factors.")

# --- User Inputs ---
st.subheader("📋 Enter Your Information")

gender = st.selectbox("Gender", ["Male", "Female"])
working_status = st.selectbox("Working Professional or Student", ["Student", "Working Professional"])

age = st.slider("Age", 10, 80, 10)
cgpa = st.slider("CGPA", 0.0, 10.0, 0.0, step=0.1)
academic_pressure = st.slider("Academic Pressure", 0, 5, 0)
work_pressure = st.slider("Work Pressure", 0, 5, 0)
study_satisfaction = st.slider("Study Satisfaction", 0, 5, 0)
job_satisfaction = st.slider("Job Satisfaction", 0, 5, 0)
sleep_duration = st.slider("Sleep Duration (hours)", 0, 12, 0)
study_hours = st.slider("Work/Study Hours", 0, 12, 0)
financial_stress = st.slider("Financial Stress", 0, 5, 0)

suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

diet_habit = st.selectbox(
    "Dietary Habits",
    ["Healthy", "Moderate", "Unhealthy"]
)

# --- Convert to numeric & scale like training ---
data = {
    "Gender": [1 if gender == "Male" else 0],
    "Age": [age / 80],  # normalized
    "Working Professional or Student": [1 if working_status == "Working Professional" else 0],
    "Academic Pressure": [academic_pressure / 5],
    "Work Pressure": [work_pressure / 5],
    "CGPA": [cgpa / 10],
    "Study Satisfaction": [study_satisfaction / 5],
    "Job Satisfaction": [job_satisfaction / 5],
    "Sleep Duration": [sleep_duration / 12],
    "Have you ever had suicidal thoughts ?": [1 if suicidal_thoughts == "Yes" else 0],
    "Work/Study Hours": [study_hours / 12],
    "Financial Stress": [financial_stress / 5],
    "Family History of Mental Illness": [1 if family_history == "Yes" else 0],
    "Dietary Habits_Moderate": [1 if diet_habit == "Moderate" else 0],
    "Dietary Habits_Unhealthy": [1 if diet_habit == "Unhealthy" else 0],
}

input_df = pd.DataFrame(data)

# Ensure all columns match model training
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# --- Prediction ---
if st.button("🔮 Predict Depression Risk"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"😔 Likely Depressed — Risk Score: {prob:.2f}")
    else:
        st.success(f"😊 Not Depressed — Risk Score: {prob:.2f}")

st.markdown("---")
st.caption("Model: XGBoost | Features excluding Profession and Degree columns")


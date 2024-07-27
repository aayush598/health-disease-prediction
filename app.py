import streamlit as st
import numpy as np
import joblib

# Load the saved models and scalers
model_files = {
    'Logistic Regression': './models/trained_model_logistic_regression.joblib',
    'SVM': './models/trained_model_svm.joblib',
    'Decision Tree': './models/trained_model_decision_tree.joblib',
    'Random Forest': './models/trained_model_random_forest.joblib',
    'KNN': './models/trained_model_knn.joblib'
}

scalers = joblib.load('./models/scalers.joblib')
models = {name: joblib.load(filename) for name, filename in model_files.items()}

# Streamlit UI for input and prediction
st.title("Heart Disease Prediction")

st.sidebar.header("Input Features")

# Function to get user input from the sidebar
def get_user_input():
    age = st.sidebar.number_input("Age", 1, 120, value=50)
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 0, 250, value=120)
    cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 0, 600, value=200)
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", ("Yes", "No"))
    resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.sidebar.number_input("Max Heart Rate", 0, 220, value=150)
    exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", ("Yes", "No"))
    oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 10.0, step=0.1, value=1.0)
    st_slope = st.sidebar.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    
    sex = 0 if sex == "Female" else 1
    chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
    fasting_bs = 0 if fasting_bs == "No" else 1
    resting_ecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exercise_angina = 0 if exercise_angina == "No" else 1
    st_slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

    features = [age, sex, chest_pain_map[chest_pain_type], resting_bp, cholesterol, fasting_bs, 
                resting_ecg_map[resting_ecg], max_hr, exercise_angina, oldpeak, st_slope_map[st_slope]]
    
    return np.array(features).reshape(1, -1)

# Get user input
input_data = get_user_input()

# Scale the input data
input_data[:, [0]] = scalers['Age'].transform(input_data[:, [0]])
input_data[:, [3]] = scalers['RestingBP'].transform(input_data[:, [3]])
input_data[:, [4]] = scalers['Cholesterol'].transform(input_data[:, [4]])
input_data[:, [7]] = scalers['MaxHR'].transform(input_data[:, [7]])
input_data[:, [9]] = scalers['Oldpeak'].transform(input_data[:, [9]])

# Display predictions for each model
st.header("Predictions")
for name, model in models.items():
    prediction = model.predict(input_data)
    st.write(f"{name}: {'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease Detected'}")

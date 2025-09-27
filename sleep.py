# %%
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Sleep Disorder Prediction App")

# Load the trained logistic regression model
try:
    with open('logreg_model.pkl', 'rb') as file:
        logreg = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'logreg_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the label encoder for decoding predictions
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Insomnia', 'Sleep Apnea', 'None'])  # Adjust based on notebook's encoding

# Sidebar for user inputs
st.sidebar.header("Enter Patient Details")

# Numerical features
age = st.sidebar.slider("Age", min_value=27, max_value=59, value=42)
sleep_duration = st.sidebar.slider("Sleep Duration (hours)", min_value=5.8, max_value=8.5, value=7.0, step=0.1)
quality_of_sleep = st.sidebar.slider("Quality of Sleep (1-10)", min_value=1, max_value=10, value=7)
physical_activity_level = st.sidebar.slider("Physical Activity Level (minutes/day)", min_value=30, max_value=90, value=60)
stress_level = st.sidebar.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
heart_rate = st.sidebar.slider("Heart Rate (bpm)", min_value=65, max_value=86, value=70)
daily_steps = st.sidebar.slider("Daily Steps", min_value=3000, max_value=10000, value=7000)
systolic_bp = st.sidebar.slider("Systolic Blood Pressure", min_value=110, max_value=150, value=128)
diastolic_bp = st.sidebar.slider("Diastolic Blood Pressure", min_value=70, max_value=100, value=85)

# Categorical features
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
occupation = st.sidebar.selectbox("Occupation", options=[
    "Software Engineer", "Doctor", "Sales Representative", "Nurse", "Engineer",
    "Accountant", "Scientist", "Lawyer", "Salesperson", "Teacher", "Manager"
])
bmi_category = st.sidebar.selectbox("BMI Category", options=["Normal", "Normal Weight", "Overweight", "Obese"])

# Function to preprocess input data
def preprocess_input(age, sleep_duration, quality_of_sleep, physical_activity_level, stress_level, 
                    heart_rate, daily_steps, systolic_bp, diastolic_bp, gender, occupation, bmi_category):
    # Create a DataFrame with numerical features
    data = {
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity_level,
        'Stress Level': stress_level,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'Systolic': systolic_bp,
        'Diastolic': diastolic_bp
    }
    df = pd.DataFrame([data])

    # One-hot encode categorical features
    df['Gender_Female'] = 1 if gender == 'Female' else 0
    df['Gender_Male'] = 1 if gender == 'Male' else 0
    
    # Initialize occupation columns
    occupations = ['Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Manager', 'Nurse', 
                   'Sales Representative', 'Salesperson', 'Scientist', 'Software Engineer', 'Teacher']
    for occ in occupations:
        df[f'Occupation_{occ}'] = 1 if occupation == occ else 0
    
    # Initialize BMI category columns
    bmi_categories = ['Normal', 'Normal Weight', 'Obese', 'Overweight']
    for bmi in bmi_categories:
        df[f'BMI Category_{bmi}'] = 1 if bmi_category == bmi else 0

    # Ensure all expected columns are present in the correct order
    expected_columns = [
        'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
        'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic', 'Gender_Female', 'Gender_Male',
        'Occupation_Accountant', 'Occupation_Doctor', 'Occupation_Engineer', 'Occupation_Lawyer',
        'Occupation_Manager', 'Occupation_Nurse', 'Occupation_Sales Representative',
        'Occupation_Salesperson', 'Occupation_Scientist', 'Occupation_Software Engineer',
        'Occupation_Teacher', 'BMI Category_Normal', 'BMI Category_Normal Weight',
        'BMI Category_Obese', 'BMI Category_Overweight'
    ]
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

# Button to make prediction
if st.sidebar.button("Predict"):
    # Preprocess the input
    input_df = preprocess_input(
        age, sleep_duration, quality_of_sleep, physical_activity_level, stress_level,
        heart_rate, daily_steps, systolic_bp, diastolic_bp, gender, occupation, bmi_category
    )
    
    # Make prediction
    try:
        prediction = logreg.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Display result
        st.subheader("Prediction Result")
        st.write(f"The predicted sleep disorder is: **{predicted_label}**")
        if predicted_label == "None":
            st.write("No sleep disorder predicted.")
        else:
            st.write(f"The patient may have {predicted_label}.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Display instructions
st.write("""
### Instructions
1. Use the sidebar to enter the patient's details.
2. Adjust the sliders for numerical features like Age, Sleep Duration, etc.
3. Select appropriate options for Gender, Occupation, and BMI Category.
4. Click the 'Predict' button to see the predicted sleep disorder.
""")



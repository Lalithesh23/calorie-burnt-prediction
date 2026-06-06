import streamlit as st
import pandas as pd
import pickle

st.title("Calories Burn Prediction")

st.markdown("Enter your details to estimate calories burned:")

gender = st.radio("Gender", ["Male", "Female"])


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, value=25)
    height = st.number_input("Height (cm)", min_value=1.0, value=170.0)
    duration = st.number_input("Exercise Duration (min)", min_value=1.0, value=30.0)

with col2:
    weight = st.number_input("Weight (kg)", min_value=1.0, value=70.0)
    heart_rate = st.number_input("Heart Rate", min_value=1.0, value=110.0)
    body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, value=37.0)

if st.button("Predict Calories Burned"):

    input_data = pd.DataFrame({
        "Gender": [gender.lower()],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp]
    })

    with open("calorie_predictor.pkl", "rb") as f:
        model = pickle.load(f)


    prediction = model.predict(input_data)

    st.success(
        f"Estimated Calories Burned: {prediction[0]:.2f} kcal"
    )
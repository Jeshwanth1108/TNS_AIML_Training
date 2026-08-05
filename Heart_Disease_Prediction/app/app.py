import streamlit as st
import pandas as pd
import joblib


# Load models

model = joblib.load(
    "../training/models/heart_model.pkl"
)


scaler = joblib.load(
    "../training/models/scaler.pkl"
)



st.title(
    "❤️ Heart Disease Prediction System"
)


st.write(
    "Enter patient details to predict heart disease risk"
)



# Input fields


age = st.number_input(
    "Age",
    min_value=1,
    max_value=100
)


sex = st.selectbox(
    "Sex",
    [
        0,
        1
    ],
    format_func=lambda x:
    "Female" if x==0 else "Male"
)


chest = st.selectbox(
    "Chest Pain Type",
    [1,2,3,4]
)


bp = st.number_input(
    "Blood Pressure"
)


cholesterol = st.number_input(
    "Cholesterol"
)


fbs = st.selectbox(
    "FBS over 120",
    [0,1]
)


ekg = st.selectbox(
    "EKG Results",
    [0,1,2]
)


max_hr = st.number_input(
    "Maximum Heart Rate"
)


exercise = st.selectbox(
    "Exercise Angina",
    [0,1]
)


st_depression = st.number_input(
    "ST Depression"
)


slope = st.selectbox(
    "Slope of ST",
    [1,2,3]
)


vessels = st.selectbox(
    "Number of vessels",
    [0,1,2,3]
)


thallium = st.selectbox(
    "Thallium",
    [3,6,7]
)



if st.button(
    "Predict"
):


    input_data = pd.DataFrame(
        [
            [
            age,
            sex,
            chest,
            bp,
            cholesterol,
            fbs,
            ekg,
            max_hr,
            exercise,
            st_depression,
            slope,
            vessels,
            thallium
            ]
        ],
        columns=[
        "Age",
        "Sex",
        "Chest pain type",
        "BP",
        "Cholesterol",
        "FBS over 120",
        "EKG results",
        "Max HR",
        "Exercise angina",
        "ST depression",
        "Slope of ST",
        "Number of vessels fluro",
        "Thallium"
        ]
    )


    scaled = scaler.transform(
        input_data
    )


    result = model.predict(
        scaled
    )


    if result[0]==1:

        st.error(
            "⚠️ High Risk: Heart Disease Presence"
        )

    else:

        st.success(
            "✅ Low Risk: No Heart Disease Detected"
        )
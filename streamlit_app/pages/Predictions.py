import streamlit as st
import pandas as pd
import numpy as np
from utils import predict_duration

def show():
    st.title("Prediction Page")
    st.markdown("Input the parameters below to predict the travel duration.")

    # Input fields
    temperature = st.number_input("Temperature (°C)", value=20.0)
    precipitation = st.number_input("Precipitation (mm)", value=0.0)
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
    cloud_cover = st.number_input("Cloud Cover (%)", value=50.0)
    visibility = st.number_input("Visibility (m)", value=10000.0)
    bias = st.number_input("Bias Column", value=1.0, disabled=True)

    # Predict button
    if st.button("Predict"):
        input_data = {
            "temperature_2m (°C)": temperature,
            "precipitation (mm)": precipitation,
            "wind_speed_10m (km/h)": wind_speed,
            "cloud_cover (%)": cloud_cover,
            "visibility (m)": visibility,
            "bias_column": bias
        }
        prediction = predict_duration(input_data)
        st.success(f"Predicted Travel Duration: {prediction / 60:.3f} minutes")
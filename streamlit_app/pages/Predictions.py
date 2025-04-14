import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import predict_duration

def show():
    st.title("Prediction Page")
    st.markdown("Input the parameters below to predict the travel duration and view the histogram.")

    # input fields
    temperature = st.number_input("Temperature (°C)", value=20.0)
    precipitation = st.number_input("Precipitation (mm)", value=0.0)
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    visibility = st.number_input("Visibility (m)", value=10000.0)

    # predict button
    if st.button("Predict"):
        # user input data
        input_data = {
            "temperature_2m (°C)": temperature,
            "precipitation (mm)": precipitation,
            "wind_speed_10m (km/h)": wind_speed,
            "relative_humidity_2m (%)": humidity,
            "visibility (m)": visibility,
            "bias": 1
        }

        # get the predicted duration from the MLP model
        prediction = predict_duration(input_data)
        st.success(f"Predicted Travel Duration: {prediction:.2f} seconds")

        # generate histogram
        st.markdown("### Predicted vs Actual Duration Histogram")

        # load actual data (y_test)
        y_test = pd.read_csv("files/ruggles2dtxg_weather_scaled.csv")["travel_time_sec"].to_numpy()

        # create a histogram for actual durations
        fig = go.Figure()

        # add histogram for actual durations
        fig.add_trace(go.Histogram(
            x=y_test,
            nbinsx=100,
            name="Actual Durations",
            marker_color="blue",
            opacity=0.7
        ))

        # add a vertical dotted line for the predicted duration
        fig.add_trace(go.Scatter(
            x=[prediction, prediction],
            y=[0, max(np.histogram(y_test, bins=100)[0]) + 5],
            mode="lines",
            name="Predicted Duration",
            line=dict(color="red", width=3, dash="dot")
        ))

        # update layout for better visualization
        fig.update_layout(
            title="Predicted vs Actual Duration",
            xaxis_title="Duration (seconds)",
            yaxis_title="Frequency",
            legend_title="Type",
            barmode="overlay"
        )

        # display the interactive plot
        st.plotly_chart(fig, use_container_width=True)
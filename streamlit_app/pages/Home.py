import streamlit as st

def show():
    st.title("Transit Duration Prediction App")
    st.markdown("""
Justin Guthrie, Edward Liu
DS4420 | Spring 2025

## The Methods:
### Baysian Modeling:
We use Bayesian modeling to predict the duration of the MBTA subway system between Ruggles and Downtown Crossing based on weather data. The model is trained on historical data and incorporates uncertainty in the predictions.

### MLP
We also use a Multi-Layer Perceptron (MLP) to predict the duration of the MBTA subway system between Ruggles and Downtown Crossing based on weather data. The MLP is trained on historical data and uses a feedforward architecture with multiple layers. We specifically index into the stations of Ruggles and Downtown Crossing as it is a common route for students looking to get into the city, and this model is represented in the app.

### Preprocessing
Preprocessing using MBTA data found here: https://mbta-massdot.opendata.arcgis.com/datasets/5f71a5c035fc4a4dad1b7fa73ba27ef8/about
- Preproc.ipynb

## App Features
- Landing page with project details (this page!).
- Prediction page for user input and delay prediction.
    """)
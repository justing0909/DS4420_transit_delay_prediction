import streamlit as st

def show():
    st.title("Transit Delay Prediction")
    st.markdown("""
# Transit Delay Prediction App
Justin Guthrie, Edward Liu
DS4420 | Spring 2025

## The Methods:
### Baysian Modeling:
We use Bayesian modeling to predict the delay of the MBTA subway system based on weather data. The model is trained on historical data and incorporates uncertainty in the predictions.

### MLP
We use a Multi-Layer Perceptron (MLP) to predict the delay of the MBTA subway system based on weather data. The MLP is trained on historical data and uses a feedforward architecture with multiple layers. We specifically index into the stations of Ruggles and Downtown Crossing as it is a common route for students looking to get into the city.

### Preprocessing
Preprocessing using MBTA data found here: https://mbta-massdot.opendata.arcgis.com/datasets/5f71a5c035fc4a4dad1b7fa73ba27ef8/about
- Preproc.ipynb

## App Features
- Landing page with project details (this page!).
- Prediction page for user input and delay prediction.
    """)
import numpy as np
import pandas as pd
from sklearn import preprocessing
from DS4420_Project_MLP import MLP, LinearAF, ReluAF

# ruggles data
df = pd.read_csv("files/ruggles2dtxg_raw.csv")[
    ['temperature_2m (°C)',
     'precipitation (mm)', 
     'relative_humidity_2m (%)', 
     'visibility (m)',
     'wind_speed_10m (km/h)']]

# scaler
def df_scaler(df,cols):
    df = df.copy(deep=True)
    scaler = preprocessing.StandardScaler()
    df.loc[:,cols] = scaler.fit_transform(df[cols])
    return df



def predict_duration(input_data):
    """
    Predicts the travel duration based on input data using the MLP model.
    
    Args:
        input_data (dict): A dictionary containing input features as keys and their values.
    
    Returns:
        float: Predicted travel duration in seconds.
    """
    # initialize the MLP model
    model = MLP(seed=102)
    model.add_layer(6, LinearAF())  # Input layer
    model.add_layer(20, LinearAF())  # Hidden layer 1
    model.add_layer(10, LinearAF())  # Hidden layer 2
    model.add_layer(1, ReluAF())  # Output layer

    # convert input_data to a numpy array
    scaled = df_scaler(pd.concat([df,
                                  pd.DataFrame(input_data,index=[0])],
                                  ignore_index = True
                                  ),
                       cols=['temperature_2m (°C)',
                             'precipitation (mm)', 
                             'relative_humidity_2m (%)', 
                             'visibility (m)',
                             'wind_speed_10m (km/h)'])
    x_scaled = scaled.to_numpy()[-1,:]

    # perform a forward pass to get the prediction
    prediction = model.fw(x_scaled)
    return prediction.item()

if __name__ == "__main__":
    print(
        predict_duration(
            input_data = {
                "temperature_2m (°C)": -10,
                "precipitation (mm)": 20,
                "wind_speed_10m (km/h)": 400,
                "relative_humidity_2m (%)": 80,
                "visibility (m)": 10000,
                "bias": 1
            }
        )
    )
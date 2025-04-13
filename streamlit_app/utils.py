import numpy as np
import pandas as pd
from DS4420_Project_MLP import MLP, LinearAF, ReluAF

def predict_duration(input_data):
    """
    Predicts the travel duration based on input data using the MLP model.
    
    Args:
        input_data (dict): A dictionary containing input features as keys and their values.
    
    Returns:
        float: Predicted travel duration in seconds.
    """
    # Initialize the MLP model
    model = MLP(seed=102)
    model.add_layer(25, LinearAF())  # Input layer
    model.add_layer(40, LinearAF())  # Hidden layer 1
    model.add_layer(80, LinearAF())  # Hidden layer 2
    model.add_layer(1, ReluAF())  # Output layer

    # Convert input_data to a numpy array
    x = np.array(list(input_data.values())).reshape(-1, 1)

    # Perform a forward pass to get the prediction
    prediction = model.fw(x)
    return prediction.item()
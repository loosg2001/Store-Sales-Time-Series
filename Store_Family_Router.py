import pandas as pd
import numpy as np

def store_family_router(X):
    """
    Creates routing keys in the format "{store_nbr}_{family}"
    Assumes X is a pandas DataFrame containing these columns.
    """
    # Cast store_nbr to int (in case it was read as float) then to string
    store_str = X['store_nbr'].astype(int).astype(str)
    
    # Cast family to string
    family_str = X['family'].astype(str)
    
    # Concatenate them with an underscore
    routes = store_str + "_" + family_str
    
    # Return as a numpy array so the custom estimator's masking logic works perfectly
    return routes.to_numpy()
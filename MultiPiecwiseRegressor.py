import numpy as np
import pandas as pd
import warnings
import time
from sklearn.base import BaseEstimator, RegressorMixin, clone
from joblib import Parallel, delayed

# --- HELPER FUNCTIONS FOR PARALLELIZATION ---
# These are kept outside the class to avoid serialization issues with joblib,
# ensuring only the strictly necessary data is sent to the worker processes.

def _fit_single_model(route_key, model, X_subset, y_subset, verbose, drop_cols):
    """
    Fits a single model on a pre-sliced subset of the training data.
    
    Args:
        route_key (str/int): The identifier for the current data route.
        model (estimator): An unfitted scikit-learn compatible estimator.
        X_subset (DataFrame/ndarray): The feature subset routed to this model.
        y_subset (Series/ndarray): The target subset routed to this model.
        verbose (int): Verbosity level.
        drop_cols (list): List of column names or indices to drop before fitting.
        
    Returns:
        tuple: (route_key, fitted_model)
    """
    if verbose != 0:
        # print(f"Training {route_key} started...")
        start_time = time.perf_counter()

    cloned_model = clone(model)
    
    # Safely handle column dropping for both Pandas DataFrames and NumPy arrays
    if drop_cols is not None:
        # Check if X_subset is a pandas DataFrame by looking for the 'drop' method
        if hasattr(X_subset, 'drop'):
            # Pandas: ignore errors if a column to drop is already missing
            X_subset = X_subset.drop(columns=drop_cols, errors='ignore')
        else:
            # NumPy: assume drop_cols contains integer column indices
            try:
                X_subset = np.delete(X_subset, drop_cols, axis=1)
            except Exception as e:
                raise ValueError(f"For NumPy arrays, 'drop_cols' must be integer indices. Error: {e}")
                
    # Fit the cloned model on the routed subset
    cloned_model.fit(X_subset, y_subset)

    # Safely extract best_estimator_ if using a grid search (e.g., GridSearchCV),
    # otherwise, just keep the standard fitted model
    final_model = getattr(cloned_model, "best_estimator_", cloned_model)

    if verbose != 0:
        end_time = time.perf_counter()
        print(f"Training {route_key} finished in {(end_time - start_time):.4f} seconds")

    return route_key, final_model


def _predict_single_model(route_key, fitted_model, X_subset, mask, verbose, drop_cols):
    """
    Generates predictions for a pre-sliced subset of data using a fitted model.
    
    Args:
        route_key (str/int): The identifier for the current data route.
        fitted_model (estimator): The trained model for this specific route.
        X_subset (DataFrame/ndarray): The feature subset routed to this model.
        mask (ndarray): Boolean array tracking the original indices of this subset.
        verbose (int): Verbosity level.
        drop_cols (list): List of column names or indices to drop before predicting.
        
    Returns:
        tuple: (route_key, mask, partial_predictions)
    """
    if verbose != 0:
        # print(f"Predicting {route_key} started...")
        start_time = time.perf_counter()

    # Apply the exact same dropping logic used during fitting
    if drop_cols is not None:
        if hasattr(X_subset, 'drop'):
            X_subset = X_subset.drop(columns=drop_cols, errors='ignore')
        else:
            X_subset = np.delete(X_subset, drop_cols, axis=1)
            
    # Generate predictions for this specific route
    partial_predictions = fitted_model.predict(X_subset)

    if verbose != 0:
        end_time = time.perf_counter()
        print(f"Predicting {route_key} finished in {(end_time - start_time):.4f} seconds")

    # Return the boolean mask alongside predictions so the main class 
    # knows exactly where to stitch these predictions back into the main array
    return route_key, mask, partial_predictions


# --- MAIN ESTIMATOR CLASS ---

class MultiPiecewiseRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible meta-estimator that partitions data based on a 
    routing condition and trains independent regressors on each partition in parallel.
    
    Attributes:
        estimators (dict): Dictionary mapping route_keys to unfitted estimator objects.
        routing_condition (callable): Function that takes X and returns an array of route_keys.
        verbose (int): Controls the verbosity of parallel operations (0 = silent).
        n_jobs (int, optional): Number of CPU cores to use. None means 1, -1 means all.
        drop_cols (list, optional): Columns to drop *after* routing but *before* fitting/predicting.
    """
    def __init__(self, estimators, routing_condition, verbose=0, n_jobs=None, drop_cols=None):
        self.estimators = estimators
        self.routing_condition = routing_condition
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.drop_cols = drop_cols 
        
    def fit(self, X, y):
        """Fits the routed models to their respective data partitions."""
        self.estimators_ = {}
        
        # Apply the routing function to determine which data goes to which model
        routes = self.routing_condition(X)
        
        # Prepare the jobs. Slicing happens HERE before joblib distribution.
        # This prevents sending the entire dataset to every CPU worker.
        fit_jobs = []
        for route_key, model in self.estimators.items():
            mask = (routes == route_key)
            
            # Only create a training job if there is actually data routed to this key
            if np.any(mask):
                fit_jobs.append(
                    delayed(_fit_single_model)(
                        route_key, model, X[mask], y[mask], self.verbose, self.drop_cols
                    )
                )
            else:
                warnings.warn(f"Warning: No training data routed to model '{route_key}'.")

        # Execute jobs in parallel across CPU cores
        if fit_jobs:
            parallel_results = Parallel(n_jobs=self.n_jobs)(fit_jobs)
            
            # Store the fitted models in the class attribute
            for route_key, fitted_model in parallel_results:
                self.estimators_[route_key] = fitted_model
                
        return self

    def predict(self, X):
        """Generates predictions by routing data to the appropriate fitted models."""
        routes = self.routing_condition(X)
        
        # Initialize the output array with NaNs instead of 0.0. 
        # This makes it easy to spot samples that never got routed to a model.
        predictions = np.full(X.shape[0], np.nan, dtype=float)
        
        # Keep track of which indices have been predicted so far
        predicted_mask = np.zeros(X.shape[0], dtype=bool)

        # Prepare prediction jobs, slicing data before sending to workers
        predict_jobs = []
        for route_key, fitted_model in self.estimators_.items():
            mask = (routes == route_key)
            
            # Only create a prediction job if there is data for this specific model
            if np.any(mask):
                predict_jobs.append(
                    delayed(_predict_single_model)(
                        route_key, fitted_model, X[mask], mask, self.verbose, self.drop_cols
                    )
                )

        # Execute prediction jobs in parallel
        if predict_jobs:
            parallel_results = Parallel(n_jobs=self.n_jobs)(predict_jobs)
            
            # Recombine partial predictions directly into the correct original indices
            for route_key, mask, partial_preds in parallel_results:
                predictions[mask] = partial_preds
                predicted_mask |= mask  # Update the master mask
                
        # Handle edge case: Warn if some data points didn't match ANY of the fitted models
        if not np.all(predicted_mask):
            unrouted_count = np.sum(~predicted_mask)
            warnings.warn(f"Warning: {unrouted_count} samples did not match any fitted model and were assigned NaN.")
            
        return predictions
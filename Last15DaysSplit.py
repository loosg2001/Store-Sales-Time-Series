import numpy as np

class Last15DaysSplit:
    """
    A custom scikit-learn compatible cross-validation splitter for sequential or time-series data.
    
    This splitter creates a single train/test split. It isolates the last `test_size` 
    samples (e.g., the last 15 days of data) to serve as the validation/test set, 
    while all preceding samples are used for the training set.
    """
    
    def __init__(self, test_size=15):
        """
        Initialize the Last15DaysSplit.
        
        Args:
            test_size (int): The number of samples from the end of the dataset 
                             to reserve for the test set. Defaults to 15.
        """
        self.test_size = test_size

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X (array-like): Training data of shape (n_samples, n_features).
            y (array-like, optional): The target variable. Defaults to None.
            groups (array-like, optional): Group labels for the samples. Defaults to None.
            
        Yields:
            tuple: A single tuple containing two NumPy arrays: (train_indices, test_indices).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Determine the actual test size to prevent indexing errors.
        # Safety check: If a specific slice of data (like a store/family) has 
        # fewer total samples than the requested test_size, ensure we leave 
        # at least 1 sample for the training set.
        actual_test_size = min(self.test_size, max(1, n_samples - 1))
        
        # Yield exactly one split of (train_indices, test_indices)
        yield indices[:-actual_test_size], indices[-actual_test_size:]

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return the number of splitting iterations in the cross-validator.
        
        Args:
            X (array-like, optional): Training data. Defaults to None.
            y (array-like, optional): The target variable. Defaults to None.
            groups (array-like, optional): Group labels. Defaults to None.
            
        Returns:
            int: The number of splits (always 1 for this specific splitter).
        """
        return 1
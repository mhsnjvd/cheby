import numpy as np

defaultTol = 1.0e-15

def iszero_numerically(v, tol=defaultTol):
    """
    Tests if the array v is numerically zero up to 
    the provided tolerance
    1.0e-15 is used if no tolerance is provided
    """

    if np.abs(v).max() < tol:
        return True
    else:
        return False

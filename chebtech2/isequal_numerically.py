import numpy as np

defaultTol = 1.0e-15

def isequal_numerically(u, v, tol=defaultTol):
    """
    Tests if the two numpy arrays u and v are 
    numerically equal up to the provided tolerance
    1.0e-15 is used if no tolerance is provided
    """
    if len(u) != len(v):
        return False
    
    d = u - v
    if np.abs(d).max() < tol:
        return True
    else:
        return False

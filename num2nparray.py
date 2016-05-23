import numpy as np

def num2nparray(x):
    """
    Convert a number or a list of numbers
    to a numpy array
    """
    #try:
    if isinstance(x, list):
        x = np.array(x)
        return x
    if isinstance(x, (int, float)):
        x = np.array([x])
        return x
    if isinstance(x, np.ndarray):
        return x

    #except TypeError:
    print('num2nparray: The input type must be int, float, list or numpy.ndarray')


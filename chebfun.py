import numpy as np
import scipy as sp

class Chebfun(object):


    #Default values:
    funHandle = lambda x: 0.0*x
    coeffs = 0.0

    # The initialser:
    def __init__(self, funHandle, coeffs):
        self.funHandle = funHandle
        self.coeffs = coeffs

    def __str__(self):
        return str(self.coeffs)
    
    __repr__ = __str__

    # Overloaded addition method:
    def __add__(self, other):
    
        newHandle = lambda x: self.funHandle(x) + other.funHandle(x)

        # Add the smaller length coeffs to larger ones:
        if ( np.shape(self.coeffs)[1] != np.shape(other.coeffs)[1]):
            print( "Houston we have a problem" )

        n = np.shape(self.coeffs)[0]       
        m = np.shape(other.coeffs)[0]

        if ( n >= m ):
            newCoeffs = self.coeffs[0, :m] + other.coeffs
        else:
            newCoeffs = self.coeffs + other.coeffs[0, :m]
            
        return Chebfun(newHandle, newCoeffs) 

    __radd__ = __add__

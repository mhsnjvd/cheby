import numpy as np
import bary
import num2nparray
from chebtech2 import vals2coeffs, coeffs2vals

class Chebtech:
    # Initialize properties of the object
    coeffs = np.array([])
    values = np.array([])
    def __init__(self, **kwargs):
        keys = kwargs.keys()
        # [TODO]if 'coeffs' in keys and 'values' in keys:
            #except 
        if 'coeffs' in keys:
            coeffs = kwargs['coeffs']
            coeffs = num2nparray.num2nparray(coeffs)
            self.coeffs = coeffs
            self.values = coeffs2vals.coeffs2vals(coeffs)
        if 'values' in keys:
            values = kwargs['values']
            values = num2nparray.num2nparray(values)
            self.values = values
            self.coeffs = vals2coeffs.vals2coeffs(values)
        if 'fun' in keys:
            self.fun = kwargs['fun']


    def length(self):
        return len(self.coeffs)

    def __getitem__(self, x):
        # Evaluate the object at the point(s) x:
        fx = bary.bary(x, self.values)
        return fx

    def sum(self):
        """
        #SUM   Definite integral of a CHEBTECH on the interval [-1,1].
        #   SUM(F) is the integral of F from -1 to 1.
        #
        #   If F is an array-valued CHEBTECH, then the result is a row vector
        #   containing the definite integrals of each column.
        #
        #   SUM(F, 2) sums over the second dimension of F, i.e., adds up its columns.
        #   If F is a scalar-valued CHEBTECH, this simply returns F.
        #
        # See also CUMSUM, DIFF.

        # Copyright 2016 by The University of Oxford and The Chebfun Developers. 
        # See http://www.chebfun.org/ for Chebfun information.
        """

        # Get the length of the values:
        # n = size(f.coeffs, 1);
        n = len(self.coeffs)

           
        if n == 0:
            # Trivial cases:
            out = 0.0
            return out
        elif n == 1:    
            # Constant CHEBTECH
            out = 2*self.coeffs
            return out

        # Evaluate the integral by using the Chebyshev coefficients (see Thm. 19.2 of
        # Trefethen, Approximation Theory and Approximation Practice, SIAM, 2013, which
        # states that \int_{-1}^1 T_k(x) dx = 2/(1-k^2) for k even):
        c = self.coeffs
        # c(2:2:end,:) = 0;
        c[1::2] = 0.0
        # out = [ 2, 0, 2./(1-(2:n-1).^2) ] * c;
        out = np.dot(np.r_[2, 0, 2/(1-np.r_[2:n]**2)], c)
        return out

    def __add__(self, other):
        result = Chebtech()
        n = self.length()
        m = other.length()
        if n >= m:
            coeffs = np.r_[other.coeffs, np.zeros(n-m)]
            result.coeffs = self.coeffs + coeffs
        else:
            coeffs = np.r_[self.coeffs, np.zeros(m-n)]
            result.coeffs = other.coeffs + coeffs

        result.values = coeffs2vals.coeffs2vals(result.coeffs)

        return result

    def __sub__(self, other):
        result = Chebtech()
        n = self.length()
        m = other.length()
        if n >= m:
            coeffs = np.r_[other.coeffs, np.zeros(n-m)]
            result.coeffs = self.coeffs - coeffs
        else:
            coeffs = np.r_[self.coeffs, np.zeros(m-n)]
            result.coeffs = other.coeffs - coeffs

        result.values = coeffs2vals.coeffs2vals(result.coeffs)

        return result

    def __str__(self):
        return "Chebtech object of length %s on [-1, 1]" % self.length()
    def __repr__(self):
        s = "Chebtech column (1 smooth piece)\n"
        s = s + "length = %s\n" % self.length()
        #return 'Chebtech object of length %s on [-1, 1]' % self.length()
        return s
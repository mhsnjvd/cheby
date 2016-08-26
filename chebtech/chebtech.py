import numpy as np
import bary
import num2nparray
import iszero_numerically
import chebtech2
import chebtech2.coeffs2vals
import chebtech2.vals2coeffs
import matplotlib.pyplot as plt

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
            self.values = chebtech2.coeffs2vals.coeffs2vals(coeffs)
        if 'values' in keys:
            values = kwargs['values']
            values = num2nparray.num2nparray(values)
            self.values = values
            self.coeffs = chebtech2.vals2coeffs.vals2coeffs(values)
        if 'fun' in keys:
            self.fun = kwargs['fun']


    def length(self):
        return len(self.coeffs)

    def __getitem__(self, x):
        # Evaluate the object at the point(s) x:
        fx = bary.bary(x, self.values)
        return fx

    def plot(self):
        x = np.linspace(-1, 1, 2001)
        plt.plot(x, self[x]) 
        plt.show()

    def isreal(self):
        if iszero_numerically.iszero_numerically(self.values.imag):
            return True
        else:
            return False

    def isimag(self):
        if iszero_numerically.iszero_numerically(self.values.real):
            return True
        else:
            return False

    def abs(self):
        #ABS   Absolute value of a CHEBTECH object.
        #   ABS(F) returns the absolute value of F, where F is a CHEBTECH 
        #   object with no roots in [-1 1]. 
        #   If ~isempty(roots(F)), then ABS(F) will return garbage
        #   with no warning. F may be complex.

        #  Copyright 2016 by The University of Oxford and The Chebfun Developers.
        #  See http://www.chebfun.org/ for Chebfun information.

        if self.isreal() or self.isimag():
            # Convert to values and then compute ABS(). 
            return Chebtech(values=np.abs(self.values))
        else:
            # [TODO]
            # f = compose(f, @abs, [], [], varargin{:});
            # [TODO]: Is the following a true copy?
            f = self
            return f

    def roots(self):
        pass

            


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


    def diff(self, order=1):
    #DIFF   Derivative of a CHEBTECH.
    #   DIFF(F) is the derivative of F and DIFF(F, K) is the Kth derivative.
    #
    # See also SUM, CUMSUM.
    
    # Copyright 2016 by The University of Oxford and The Chebfun Developers. 
    # See http://www.chebfun.org/ for Chebfun information.
    
    ################################################################################
    # If the CHEBTECH G of length n is represented as
    #       \sum_{r=0}^{n-1} b_r T_r(x)
    # its derivative is represented with a CHEBTECH of length n-1 given by
    #       \sum_{r=0}^{n-2} c_r T_r(x)
    # where c_0 is determined by
    #       c_0 = c_2/2 + b_1;
    # and for r > 0,
    #       c_r = c_{r+2} + 2*(r+1)*b_{r+1},
    # with c_n = c_{n+1} = 0.
    #
    # [Reference]: Page 34 of Mason & Handscomb, "Chebyshev Polynomials". Chapman &
    # Hall/CRC (2003).
    ################################################################################
    
    ## Check the inputs:
    
    n = len(self.coeffs)

    # Trivial case of an empty CHEBTECH:
    if n == 0:
        return self.copy()
    
    if order == 0:
        return self.copy()

    assert(order>0)
    # Differentiate with respect to the continuous variable by default:

    if ( nargin < 3 )
        dim = 1;
    end
    
    if ( dim == 1 )
        # Take difference across 1st dimension:
        f = diffContinuousDim(f, k);
    else
        # Take difference across 2nd dimension:
        f = diffFiniteDim(f, k);
    end
    
    end
    
    function f = diffFiniteDim(f, k)
    # Take kth difference across 2nd dimension (i.e., across columns).
    
        if ( k >= size(f, 2) )
            # The output will be an empty CHEBTECH:
            f = f.make();
        else 
            # Differentiate coefficients across columns:
            f.coeffs = diff(f.coeffs, k, 2);
        end
    end
    
    function f = diffContinuousDim(f, k)
    # Differentiate in the first dimension (i.e., df/dx).
        
        # Get the coefficients:
        c = f.coeffs;
    
        # Get their length:
        n = size(c, 1);
    
        # If k >= n, we know the result will be the zero function:
        if ( k >= n ) 
            f = f.make(zeros(1, size(f, 2)));
            return
        end
        
        # Loop for higher derivatives:
        for m = 1:k
            # Compute new coefficients using recurrence:
            c = computeDerCoeffs(c);
            n = n - 1;
        end
        
        # Store new coefficients:
        f.coeffs = c;
        
    end
          
    function cout = computeDerCoeffs(c)
    #COMPUTEDERCOEFFS   Recurrence relation for coefficients of derivative.
    #   C is the matrix of Chebyshev coefficients of a (possibly array-valued)
    #   CHEBTECH object.  COUT is the matrix of coefficients for a CHEBTECH object
    #   whose columns are the derivatives of those of the original.
    
        [n, m] = size(c);
        cout = zeros(n-1, m);                        # Initialize vector {c_r}
        w = repmat(2*(1:n-1)', 1, m);
        v = w.*c(2:end,:);                           # Temporal vector
        cout(n-1:-2:1,:) = cumsum(v(n-1:-2:1,:), 1); # Compute c_{n-2}, c_{n-4}, ...
        cout(n-2:-2:1,:) = cumsum(v(n-2:-2:1,:), 1); # Compute c_{n-3}, c_{n-5}, ...
        cout(1,:) = .5*cout(1,:);                    # Adjust the value for c_0
    end
















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

        result.values = chebtech2.coeffs2vals.coeffs2vals(result.coeffs)

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

        result.values = chebtech2.coeffs2vals.coeffs2vals(result.coeffs)

        return result

    def __str__(self):
        return "Chebtech object of length %s on [-1, 1]" % self.length()
    def __repr__(self):
        s = "Chebtech column (1 smooth piece)\n"
        s = s + "length = %s\n" % self.length()
        #return 'Chebtech object of length %s on [-1, 1]' % self.length()
        return s

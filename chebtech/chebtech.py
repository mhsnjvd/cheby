import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import copy

def iszero_numerically(v, tol=None):
    """
    Tests if the array v is numerically zero up to 
    the provided tolerance in infinity norm.
    1.0e-15 is used if no tolerance is provided
    """
    default_tol = 1.0e-15

    if tol is None:
        tol = default_tol

    if np.abs(v).max() < tol:
        return True
    else:
        return False


def refine_by_resampling(op, values):
    #REFINERESAMPLING   Default refinement function for resampling scheme.

    min_samples = 2**4 + 1
    max_samples = 2**16 + 1 
    extrapolate_flag = False


    if values.size == 0:
        # Choose initial n based upon min_samples:
        n = 2 ** np.ceil(np.log2(min_samples - 1)) + 1
    else:
        # (Approximately) powers of sqrt(2):
        power = np.log2(len(values) - 1)
        if ( (power == np.floor(power)) and (power > 5) ):
            n = np.round(2**(np.floor(power) + .5)) + 1
            n = n - mod(n, 2) + 1
        else:
            n = 2**(np.floor(power) + 1) + 1
    
    # n is too large:
    if ( n > max_samples ):
        # Don't give up if we haven't sampled at least once.
        if ( values.size == 0 ):
            n = max_samples
            give_up = False
        else:
            give_up = True
            return values, give_up
    else:
        give_up = False;
   
    # 2nd-kind Chebyshev grid:
    x = Chebtech.chebpts(n);

    # Evaluate the operator:
    if ( extrapolate_flag ):
        values = np.r_[np.nan, op(x[1:-1]), np.nan]
    else:
        values = op(x)
    
    return values, give_up

def refine_by_nesting(op, values):
    """REFINENESTED  Default refinement function for single ('nested') sampling."""

    min_samples = 2**4 + 1
    max_samples = 2**16 + 1

    if ( values.size == 0 ):
        # The first time we are called, there are no values
        # and REFINENESTED is the same as REFINERESAMPLING.
        values, give_up = refine_by_esampling(op, values, pref);
    else:
        # Compute new n by doubling (we must do this when not resampling).
        n = 2*len(values) - 1;
        
        # n is too large:
        if ( n > max_samples ):
            give_up = True
            return values, give_up
        else:
            give_up = False
        
        # 2nd-kind Chebyshev grid:
        x = Chebtech.chebpts(n);
        # Take every 2nd entry:
        # x = x(2:2:end-1);
        x = x[1:-1:2]

        # Shift the stored values:
        # values(1:2:n,:) = values;
        values[:n:2] = values
        # Compute and insert new ones:
        # values(2:2:end-1,:) = feval(op, x);
        values[1::2] = op(x)
        return values, give_up

class Chebtech:
    # Initialize properties of the object
    #default_dtype = np.float64
    default_dtype = np.complex128
    
    def __init__(self, fun=None, **kwargs):
        self.coeffs = np.array([], dtype=type(self).default_dtype)
        self.ishappy = False

        keys = kwargs.keys()
        # [TODO]if 'coeffs' in keys and 'values' in keys:
            #except 
        if 'coeffs' in keys:
            coeffs = np.asarray(kwargs['coeffs'], dtype=type(self).default_dtype)
            self.coeffs = coeffs.copy()
            self.ishappy = True
        if 'values' in keys:
            values = np.asarray(kwargs['values'], dtype=type(self).default_dtype)
            self.coeffs = Chebtech.vals2coeffs(values)
            self.ishappy = True
        if fun is not None:
            self.fun = fun
            self.ishappy = False
            self.__ctor__(fun)

    def __ctor__(self, f):
        """Construct a chebtech object from a function f."""
        n_max = 65537
        tail_length = 5
        tol = 1.0e-15
        self.ishappy = False
        n = 9
        while ( not self.ishappy and n <= n_max):
            x = Chebtech.chebpts(n)
            values = f(x)
            coeffs = Chebtech.vals2coeffs(values)
            if np.sum(np.abs(coeffs[-tail_length:])) < tail_length * tol:
                self.ishappy = True
                # Trim zeros from the back
                coeffs_mask = np.nonzero(np.abs(coeffs) > tol)[0]
                if len(coeffs_mask) == 0:
                    self.coeffs = Chebtech.zeros(1)
                    return
                else:
                    n = coeffs_mask[-1] + 1
                    self.coeffs = coeffs[:n]
                    return
            else:
                n = 2*n
        print('Function did not converge')
        self.ishappy = False

    def values(self):
        return Chebtech.coeffs2vals(self.coeffs)

    def length(self):
        return len(self.coeffs)

    def points(self):
        n = self.length()
        if n == 0:
            return Chebtech.empty_array()
        else:
            return Chebtech.chebpts(n)

    def vscale(self):
        if self.length() == 0:
            return np.nan
        else:
            return np.max(np.abs(self.values()))

    def plot(self):
        x = np.linspace(-1, 1, 2001)
        plt.plot(x, self[x]) 
        plt.show()

    def isreal(self):
        if iszero_numerically(self.values().imag):
            return True
        else:
            return False

    def isimag(self):
        if iszero_numerically(self.values().real):
            return True
        else:
            return False

    def isequal(self, other):
        if not isinstance(other, Chebtech):
            # [TODO] something must be done
            print('isequal() accepts a Chebtech object only.')
        
        # Get coefficients and trim zeros at the end
        a = np.trim_zeros(self.coeffs, 'b')
        b = np.trim_zeros(self.coeffs, 'b')
            
        # compare coefficients
        if len(a) != len(b):
            return False
        elif np.all(a==b):
            return True
        else:
            return False

    def __eq__(self, other):
        return self.isequal(other)

    def real(self):
        """Real part of a CHEBTECH."""

        # Compute the real part of the coefficients:
        c = self.coeffs.real

        if not np.any(c):
            # [TODO] we need a tolerance here on c?
            # Input was purely imaginary, so output a zero CHEBTECH:
            return Chebtech(coeffs=Chebtech.zeros(1))
        else:
            return Chebtech(coeffs=c)

    def imag(self):
        """Imaginary part of a Chebtech."""

        # Compute the imaginary part of the coefficients:
        c = self.coeffs.imag

        if not np.any(c):
            # [TODO] we need a tolerance here on c?
            # Input was purely real, so output a zero CHEBTECH:
            return Chebtech(coeffs=Chebtech.zeros(1))
        else:
            return Chebtech(coeffs=c)

    def conjugate(self):
        """Conjugate of a Chebtech."""
        if self.isreal():
            return copy.deepcopy(self)
        else:
            coeffs = np.conjugate(self.coeffs)
            return Chebtech(coeffs=coeffs)

    def conj(self):
        """Alias of conjugate"""
        return self.conjugate()

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
            return Chebtech(values=np.abs(self.values()))
        else:
            # [TODO]
            # f = compose(f, @abs, [], [], varargin{:});
            # [TODO]: Is the following a true copy?
            f = self
            return f

    def roots(self, **kwargs):
        """Roots of a CHEBTECH in the interval [-1,1].
        """

        def roots_main(c, htol):
            """Compute the roots of the polynomial given by the coefficients c on the unit interval."""
       
            # # Define these as persistent, need to compute only once.
            # persistent Tleft Tright
       
            # Simplify the coefficients:
            tail_max = np.spacing(1) * np.abs(c).sum()
            # # Find the final coefficient about tailMax:
            big_coeffs_mask = np.nonzero(np.abs(c) > tail_max)[0]
            if len(big_coeffs_mask) == 0:
                n = 0
            else:
                n = big_coeffs_mask[-1] + 1

            # # Truncate the coefficients (rather than alias):
            if n > 1 and n < len(c):
                c = c[:n]
       
            max_eig_size = 50

            if n == 0:
                if roots_pref['zero_fun']:
                    r = Chebtech.zeros(1)
                else:
                    r = Chebtech.empty_array()
            elif n == 1:
                if c[0] == 0.0 and roots_pref['zero_fun']:
                    r = Chebtech.zeros(1)
                else:
                    r = Chebtech.empty_array()
            elif n == 2:
                r = -c[0]/c[1]
                if not roots_pref['all']:
                    if np.abs(r.imag) > htol or r < -(1 + htol) or r > (1 + htol):
                        r = Chebtech.empty_array()
                    else:
                        r = np.min([r.real, 1])
                        r = np.max([r, -1])
            elif not roots_pref['recurse'] or n <= max_eig_size:
                c_old = np.copy(c)
                c = -0.5 * c[:-1]/c[-1]
                c[-2] = c[-2] + 0.5

                oh = 0.5 * Chebtech.ones(len(c)-1)
                A = np.diag(oh, 1) + np.diag(oh, -1)
                A[-2, -1] = 1.0
                A[:, 0] = np.flipud(c)

                if roots_pref['qz']: 
                    B = Chebtech.eye(A.shape[0], A.shape[1])
                    c_old = c_old / np.abs(c_old).max()
                    B[0, 0] = c_old[-1]
                    c_old = -0.5 * c_old[:-1]
                    c_old[-2] = c_old[-2] + 0.5 * B[0, 0]
                    A[:, 0] = np.flipud(c_old)
                    r = linalg.eig(A, b=B)[0]
                else:
                    r = linalg.eig(A)[0]

                # Clean the roots up a bit:
                if not roots_pref['all']: 
                    # Remove dangling imaginary parts:
                    mask = np.abs(r.imag) < htol
                    r = r[mask].real
                    # Keep roots inside [-1 1]:
                    r = np.sort(r[np.abs(r) <= 1 + htol] );
                    # Correct roots over ends:
                    if len(r) != 0:
                        r[0] = np.max([r[0], -1])
                        r[-1] = np.min([r[-1], 1])
                elif roots_pref['prune']:
                    rho = np.sqrt(np.spacing(1))**(-1/n);
                    rho_roots = np.abs(r + np.sqrt(r**2 - 1))
                    rho_roots[rho_roots < 1] = 1.0/rho_roots[rho_roots < 1]
                    r = r[rho_roots <= rho]
            # Otherwise, split using more traditional methods (i.e., Clenshaw):
            else:
                # Evaluate the polynomial on both intervals:
                x_left = chebptsAB(n, [ -1, split_point])
                x_right = chebptsAB(n, [split_point, 1])
                xx = np.r_[x_left, x_right]
                v = Chebtech.clenshaw(xx, c);
       
                # Get the coefficients on the left:
                c_left = Chebtech.vals2coeffs(v[:n])
       
                # Get the coefficients on the right:
                c_right = Chebtech.vals2coeffs(v[n:])
       
                # Recurse:
                r_left = roots_main(c_left, 2*htol)
                r_right = roots_main(c_right, 2*htol)
                r1 = (split_point - 1.0)/2.0 + (split_point + 1.0)/2.0 * r_left
                r2 = (split_point + 1.0)/2.0 + (1.0 - split_point)/2.0 * r_right
                r = np.r_[r1, r2]

            return np.r_[r]

       
        def chebptsAB(n, ab):
            """chebpts in an interval."""
            a = ab[0]
            b = ab[1]
            x = Chebtech.chebpts(n)
            y = b*(x + 1)/2 + a*(1 - x)/2
            return y
    
        # # Deal with empty case:
        # if ( isempty(f) )
        #     out = [];
        #     return
        # end

        if len(self.coeffs) == 0:
            return Chebtech.empty_array()
        
        # Default preferences:
        roots_pref = {}
        roots_pref['all'] = kwargs.setdefault('all', False)
        roots_pref['complex_roots'] = kwargs.setdefault('complex_roots', False)
        roots_pref['recurse'] = kwargs.setdefault('recurse', True)
        roots_pref['prune'] = kwargs.setdefault('prune', False)
        roots_pref['zero_fun'] = kwargs.setdefault('zero_fun', True)
        roots_pref['qz'] = kwargs.setdefault('qz', False)
        roots_pref['filter'] = kwargs.setdefault('filter', None)

        if roots_pref['complex_roots']:
            roots_pref['all'] = True
            roots_pref['prune'] = True

        # Subdivision maps [-1,1] into [-1, split_point] and [split_point, 1].
        # This is an arbitrary number.
        split_point = -0.004849834917525


        # Trivial case for f constant:
        if self.length() == 1 or self.vscale() == 0.0:
            if self.coeffs[0] == 0.0 and roots_pref['zero_fun']:
                # Return a root at centre of domain:
                out = Chebtech.zeros(1)
            else:
                # Return empty:
                out = Chebtech.empty_array()
            return out

        # Get scaled coefficients for the recursive call:
        c = np.copy(self.coeffs)/self.vscale()

        # Call the recursive rootsunit function:
        # TODO:  Does the tolerance need to depend on some notion of hscale?
        default_tol = np.spacing(1)*100
        r = roots_main(c, default_tol)

        # [TODO] Try to filter out spurious roots:
        if roots_pref['filter'] is not None:
            # r = np.sort(r, 'ascend');
            # r = np.sort(r, 'ascend');
            # fltr = roots_pref['filter']
            # r = fltr(r, f)
            pass

        # Prune the roots, if required:
        if roots_pref['prune'] and not roots_pref['recurse']:
            rho = np.sqrt(np.spacing(1))**(-1/self.length())
            rho_roots = np.abs(r + np.sqrt(r**2 - 1))
            rho_roots[rho_roots < 1] = 1/rho_roots[rho_roots < 1]
            out = r[rho_roots <= rho]
        else:
            out = r

        return out

        

    def cumsum(self):
        """Indefinite integral of a CHEBTECH.
        #   CUMSUM(F) is the indefinite integral of the CHEBTECH F with the constant of
        #   integration chosen so that F(-1) = 0.
        #
        #   CUMSUM(F, 2) will take cumulative sum over the columns of F which is an
        #   array-valued CHEBTECH.
        #
        # See also DIFF, SUM.
        
        # Copyright 2016 by The University of Oxford and The Chebfun Developers.
        # See http://www.chebfun.org/ for Chebfun information.
        
        ###########################################################################
        # If the CHEBTECH G of length n is represented as
        #       \sum_{r=0}^{n-1} c_r T_r(x)
        # its integral is represented with a CHEBTECH of length n+1 given by
        #       \sum_{r=0}^{n} b_r T_r(x)
        # where b_0 is determined from the constant of integration as
        #       b_0 = \sum_{r=1}^{n} (-1)^(r+1) b_r;
        # and other coefficients are given by
        #       b_1 = c_0 - c_2/2,
        #       b_r = (c_{r-1} - c_{r+1})/(2r) for r > 0,
        # with c_{n+1} = c_{n+2} = 0.
        #
        # [Reference]: Pages 32-33 of Mason & Handscomb, "Chebyshev Polynomials".
        # Chapman & Hall/CRC (2003).
        ###########################################################################
        """

        # # Initialise storage:
        # c = f.coeffs;                      # Obtain Chebyshev coefficients {c_r}
        c = self.coeffs.copy()

        n = len(c)

        if n == 0:
            return Chebtech()
        
        # c = [ c ; zeros(2, m) ;];          # Pad with zeros
        c = np.r_[c, Chebtech.zeros(2)]
        # b = zeros(n+1, m);                 # Initialize vector b = {b_r}
        b = Chebtech.zeros(n+1)
        
        # # Compute b_(2) ... b_(n+1):
        # b(3:n+1,:) = (c(2:n,:) - c(4:n+2,:)) ./ repmat(2*(2:n).', 1, m);
        b[2:] = (c[1:n] - c[3:n+2])/(2*np.r_[2:n+1])
        # b(2,:) = c(1,:) - c(3,:)/2;        # Compute b_1
        b[1] = c[0] - c[2]/2
        # v = ones(1, n);
        v = Chebtech.ones(n)
        # v(2:2:end) = -1;
        v[1::2] = -1.0
        # b(1,:) = v*b(2:end,:);             # Compute b_0 (satisfies f(-1) = 0)
        b[0] = np.dot(v, b[1:])

        return Chebtech(coeffs=b)
        
        
        # [TODO]
        # # Simplify (as suggested in Chebfun ticket #128)
        # f = simplify(f);
        
        # # Ensure f(-1) = 0:
        # lval = get(f, 'lval');
        # f.coeffs(1,:) = f.coeffs(1,:) - lval;
        

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

        c = self.coeffs.copy()

        # Get the length of the coefficients:
        n = len(c)

        if n == 0:
            # Trivial cases:
            out = 0.0
            return out
        elif n == 1:    
            # Constant CHEBTECH
            out = 2*c
            return out

        # Evaluate the integral by using the Chebyshev coefficients (see Thm. 19.2 of
        # Trefethen, Approximation Theory and Approximation Practice, SIAM, 2013, which
        # states that \int_{-1}^1 T_k(x) dx = 2/(1-k^2) for k even):
        # c(2:2:end,:) = 0;
        c[1::2] = 0.0
        # out = [ 2, 0, 2./(1-(2:n-1).^2) ] * c;
        out = np.dot(np.r_[2, 0, 2/(1-np.r_[2:n]**2)], c)
        return out


    def diff(self, order=1):
        """Derivative of a CHEBTECH.
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
        """

        def computeDerCoeffs(coeffs):
        #COMPUTEDERCOEFFS   Recurrence relation for coefficients of derivative.
        #   C is the matrix of Chebyshev coefficients of a (possibly array-valued)
        #   CHEBTECH object.  COUT is the matrix of coefficients for a CHEBTECH object
        #   whose columns are the derivatives of those of the original.
            
            c = np.copy(coeffs)

            n = len(c)
            if n <= 1:
                return Chebtech.zeros(1)


            cout = Chebtech.zeros(n-1)
            #w = repmat(2*(1:n-1)', 1, m);
            w = 2*np.r_[1:n]
            #v = w.*c(2:end,:);                           # Temporal vector
            v = w*c[1:]
            #cout(n-1:-2:1,:) = cumsum(v(n-1:-2:1,:), 1); # Compute c_{n-2}, c_{n-4}, ...
            cout[n-2::-2] = v[n-2::-2].cumsum() # Compute c_{n-2}, c_{n-4}, ...
            #cout(n-2:-2:1,:) = cumsum(v(n-2:-2:1,:), 1); # Compute c_{n-3}, c_{n-5}, ...
            cout[n-3::-2] = v[n-3::-2].cumsum() # Compute c_{n-2}, c_{n-4}, ...
            #cout(1,:) = .5*cout(1,:);                    # Adjust the value for c_0
            cout[0] = 0.5 * cout[0]
            return cout

    
        ## Check the inputs:
        n = len(self.coeffs)

        assert(order == np.round(order) and order >= 0)

        # Trivial case of an empty CHEBTECH:
        if n == 0:
            return np.copy(self)
        
        if order == 0:
            return np.copy(self)

        # Get the coefficients:
        c = np.copy(self.coeffs)
        
        # If k >= n, we know the result will be the zero function:
        if ( order >= n ):
            return Chebtech(coeffs=np.array([0.0]))
            
        # Loop for higher derivatives:
        for m in range(0, order):
            # Compute new coefficients using recurrence:
            c = computeDerCoeffs(c);

        # Return a Chebtech made of the new coefficients:
        return Chebtech(coeffs=c)


    def minandmax(self):
        """Global minimum and maximum on [-1,1].
        #   VALS = MINANDMAX(F) returns a 2-vector VALS = [MIN(F); MAX(F)] with the
        #   global minimum and maximum of the CHEBTECH F on [-1,1].  If F is a
        #   array-valued CHEBTECH, VALS is a 2-by-N matrix, where N is the number of
        #   columns of F.  VALS(1, K) is the global minimum of the Kth column of F on
        #   [-1, 1], and VALS(2, K) is the global maximum of the same.
        #
        #   [VALS, POS] = MINANDMAX(F) returns also the 2-vector POS where the minimum
        #   and maximum of F occur.
        #
        #   If F is complex-valued the absolute values are taken to determine extrema
        #   but the resulting values correspond to those of the original function. That
        #   is, VALS = FEVAL(F, POS) where [~, POS] = MINANDMAX(ABS(F)). (In fact,
        #   MINANDMAX actually computes [~, POS] = MINANDMAX(ABS(F).^2), to avoid
        #   introducing singularities to the function).
        #
        # See also MIN, MAX.
    
        # Copyright 2016 by The University of Oxford and The Chebfun Developers.
        # See http://www.chebfun.org/ for Chebfun information.
        """
    
        if not self.isreal():
            # We compute sqrt(max(|f|^2))to avoid introducing a singularity.
            realf = self.real()
            imagf = self.imag()
            h = realf*realf + imagf*imagf
            h = h.simplify()
            ignored, pos = h.minandmax()
            vals = self[pos]
            return vals, pos
    
        # Compute derivative:
        fp = self.diff()
    
        # Make the Chebyshev grid (used in minandmaxColumn).
        xpts = self.points()
    
    
        # Initialise output
        pos = Chebtech.zeros(2)
        vals = Chebtech.zeros(2)
        
        if self.length() == 1:
            vals = self[pos]
            return vals, pos
        
        # Compute critical points:
        r = fp.roots()
        r = np.unique(np.r_[-1.0, r, 1.0])
        v = self[r]
    
        # min
        vals[0] = np.min(v)
        pos[0] = r[np.argmin(v)]
    
        # Take the minimum of the computed minimum and the function values:
        values = Chebtech.coeffs2vals(self.coeffs)
        temp = np.r_[vals[0], values]
        vmin = np.min(temp)
        vindex = np.argmin(temp)
        if vmin < vals[0]:
            vals[0] = vmin;
            pos[0] = xpts[vindex - 1]
    
        # max
        vals[1] = np.max(v)
        pos[1] = r[np.argmax(v)]
    
        # Take the maximum of the computed maximum and the function values:
        temp = np.r_[vals[1], values]
        vmax = np.max(temp)
        vindex = np.argmax(temp)
        if vmax > vals[1]:
            vals[1] = vmax
            pos[1] = xpts[vindex - 1]
    
        return vals, pos

    def max(self):
        """Global maximum of a CHEBTECH on [-1,1]."""

        minmax, pos = self.minandmax()
        return minmax[1]

    def argmax(self):
        """Location of global maximum of a CHEBTECH on [-1,1]."""

        minmax, pos = self.minandmax()
        return pos[1]

    def min(self):
        """Global minimum of a CHEBTECH on [-1,1]."""

        minmax, pos = self.minandmax()
        return minmax[0]

    def argmin(self):
        """Location of global minimum of a CHEBTECH on [-1,1]."""

        minmax, pos = self.minandmax()
        return pos[0]

    def simplify(self):
        return copy.deepcopy(self)

    def flipud(self):
        """FLIPUD   Flip/reverse a CHEBTECH object.
           G = FLIPUD(F) returns G such that G(x) = F(-x) for all x in [-1,1].
        """

        # Negate the odd coefficients:
        self.coeffs[1::2] = -self.coeffs[1::2]
        # Note: the first odd coefficient is at index 2.


    def __add__(self, other):
        if not isinstance(other, Chebtech):
            return self.__radd__(other)

        result = Chebtech()
        n = self.length()
        m = other.length()
        if n == 0 or m == 0:
            return result
        if n >= m:
            coeffs = np.r_[other.coeffs, Chebtech.zeros(n-m)]
            result.coeffs = self.coeffs + coeffs
        else:
            coeffs = np.r_[self.coeffs, Chebtech.zeros(m-n)]
            result.coeffs = other.coeffs + coeffs

        result.ishappy = self.ishappy and other.ishappy

        return result


    def __sub__(self, other):
        if not isinstance(other, Chebtech):
            return self.__rsub__(other)

        result = Chebtech()
        n = self.length()
        m = other.length()
        if n == 0 or m == 0:
            return result
        if n >= m:
            coeffs = np.r_[other.coeffs, Chebtech.zeros(n-m)]
            result.coeffs = self.coeffs - coeffs
        else:
            coeffs = np.r_[self.coeffs, Chebtech.zeros(m-n)]
            result.coeffs = coeffs - other.coeffs

        result.ishappy = self.ishappy and other.ishappy

        return result

    def __radd__(self, other):
        result = Chebtech()
        n = self.length()
        if n == 0:
            result = Chebtech()
        else:
            result.coeffs = np.copy(self.coeffs)
            result.coeffs[0] = result.coeffs[0] + other

        result.ishappy = self.ishappy

        return result

    def __rsub__(self, other):
        return self.__radd__(-1.0*other)

    def __mul__(self, other):
        if not isinstance(other, Chebtech):
            return self.__rmul__(other)

        n = len(self.coeffs)
        m = len(other.coeffs)

        if ( n == 0 or m == 0): # Empty cases
            return Chebtech()
        elif ( n == 1): # Constant case
            return other.__rmul__(self.coeffs[0])
        elif ( m == 1): # Constant case
            return self.__rmul__(other.coeffs[0])
        else: # General case
            fc = np.r_[ self.coeffs[:], Chebtech.zeros(m+1)]
            gc = np.r_[other.coeffs[:], Chebtech.zeros(n+1)]

            #mn = size(fc, 1);
            # N = m + n + 1
            N = len(fc)
            #t = [2*fc(1,:) ; fc(2:end,:)];                    % Toeplitz vector.
            t = np.r_[2.0*fc[0], fc[1:]]
            #x = [2*gc(1,:) ; gc(2:end,:)];                    % Embed in Circulant.
            x = np.r_[2.0*gc[0], gc[1:]]
            #xprime = fft([x ; x(end:-1:2,:)]);                % FFT for Circulant mult.
            xprime = np.fft.fft(np.r_[x, x[-1:0:-1]])
            #aprime = fft([t ; t(end:-1:2,:)]);
            aprime = np.fft.fft(np.r_[t, t[-1:0:-1]])
            #Tfg = ifft(aprime.*xprime);                   % Diag in function space.
            Tfq = np.fft.ifft(aprime * xprime)
            #hc = .25*[Tfg(1,:); Tfg(2:end,:) + Tfg(end:-1:2,:)];% Extract out result.
            out_coeffs = 0.25 * np.r_[Tfq[0], Tfq[1:] + Tfq[-1:0:-1]]
            #hc = hc(1:mn,:);                                % Take the first half.
            out_coeffs = out_coeffs[:N]

            result = Chebtech(coeffs=out_coeffs)

            # Check for two cases where the output is known in advance to be positive,
            # namely F == conj(G) or F == G and isreal(F).
            result_is_positive = (np.all(self.coeffs == other.coeffs) and self.isreal()) or (np.all(np.conjugate(self.coeffs) == other.coeffs))

            #% [TODO] Update ishappy:
            #f.ishappy = f.ishappy && g.ishappy;

            # Simplify!
            result = result.simplify()

            if result_is_positive:
                # Here we know that the product of F and G should be positive. However,
                # SIMPLIFY may have destroyed this property, so we enforce it.
                values = Chebtech.coeffs2vals(result.coeffs)
                values = np.abs(values)
                result.coeffs = Chebtech.vals2coeffs(values)
            return result


    def __rmul__(self, other):
        result = Chebtech()
        n = self.length()
        if n == 0:
            result = Chebtech()
        else:
            result.coeffs = self.coeffs * other

        return result

    def __len__(self):
        return self.length()

    def __str__(self):
        return "Chebtech object of length %s on [-1, 1]" % self.length()

    def __repr__(self):
        s = "Chebtech column (1 smooth piece)\n"
        s = s + "length = %s\n" % self.length()
        #return 'Chebtech object of length %s on [-1, 1]' % self.length()
        return s
    def __call__(self, x):
        return self[x]

    def __getitem__(self, x):
        # Evaluate the object at the point(s) x:
        # fx = bary.bary(x, self.values)
        fx = Chebtech.clenshaw(x, self.coeffs)
        return fx

    @staticmethod
    def empty_array():
        return np.array([], dtype=Chebtech.default_dtype)

    @staticmethod
    def zeros(n):
        return np.zeros(n, dtype=Chebtech.default_dtype)

    @staticmethod
    def ones(n):
        return np.ones(n, dtype=Chebtech.default_dtype)

    @staticmethod
    def eye(*args):
        return np.eye(*args, dtype=Chebtech.default_dtype)

    @staticmethod
    def chebpts(n):
        """CHEBPTS   Chebyshev points in [-1, 1]."""

        # Special case (no points)
        if n == 0:
            x = Chebtech.empty_array()
        # Special case (single point)
        elif n == 1:
            x = Chebtech.zeros(1)
        else:
            # Chebyshev points:
            m = n - 1
            # (Use of sine enforces symmetry.)
            x = np.sin(np.pi*(np.r_[-m:m+1:2]/(2*m)))
        return x

    @staticmethod
    def vals2coeffs(values):
        """Convert values at Chebyshev points to Chebyshev coefficients.
    
        #   C = VALS2COEFFS(V) returns the (N+1)x1 vector of coefficients such that F(x)
        #   = C(1)*T_0(x) + C(2)*T_1(x) + C(N+1)*T_N(x) (where T_k(x) denotes the
        #   k-th 1st-kind Chebyshev polynomial) interpolates the data [V(1) ; ... ;
        #   V(N+1)] at Chebyshev points of the 2nd kind.
        #
        #   Input: values must be of type numpy.ndarray
        #   Output: a numpy.ndarray of the same size as values
        #
        # See also COEFFS2VALS, CHEBPTS.
    
        # Copyright 2016 by The University of Oxford and The Chebfun Developers. 
        # See http://www.chebfun.org/ for Chebfun information.
    
        ################################################################################
        # [Developer Note]: This is equivalent to the Inverse Discrete Cosine Transform
        # of Type I.
        #
        # [Mathematical reference]: Section 4.7 Mason & Handscomb, "Chebyshev
        # Polynomials". Chapman & Hall/CRC (2003).
        ################################################################################
        """
    
        # *Note about symmetries* The code below takes steps to 
        # ensure that the following symmetries are enforced:
        # VALUES exactly even ==> odd Chebyshev COEFFS are exactly zero
        # VALUES exactly odd ==> even Chebyshev COEFFS are exactly zero
        # These corrections are required because the MATLAB FFT does not
        # guarantee that these symmetries are enforced.
    
        # Make sure everything is in floating point:
        values = 1.0 * values

        # Get the length of the input:
        n = len(values)
    
        # Trivial case (constant):
        if n <= 1:
            coeffs = np.copy(values)
            return coeffs

        # check for symmetry
        iseven = np.max(np.abs(values-np.flipud(values))) == 0.0
        isodd = np.max(np.abs(values+np.flipud(values))) == 0.0
    
        # Mirror the values (to fake a DCT using an FFT):
        tmp = np.r_[values[n-1:0:-1], values[0:n-1]]
    
        if np.isreal(values).all():
            # Real-valued case:
            coeffs = np.fft.ifft(tmp)
            coeffs = coeffs.real
        elif np.isreal(1.0j*values).all():
            # Imaginary-valued case:
            coeffs = np.fft.ifft(tmp.imag)
            coeffs = 1.0j*coeffs.real
        else:
            # General case:
            coeffs = np.fft.ifft(tmp)
    
        # Truncate:
        coeffs = coeffs[0:n]
    
        # Scale the interior coefficients:
        coeffs[1:n-1] = 2.0*coeffs[1:n-1]

    
        # adjust coefficients for symmetry
        # [TODO] Is the np.fft already symmetric? in which 
        # case we don't need this extra enforcing
        if iseven:
            coeffs[1::2] = 0.0
        if isodd:
            coeffs[::2] = 0.0
    
        return (1.0 + 0.0j) * coeffs
    
    
    
    @staticmethod
    def coeffs2vals(coeffs):
        """Convert Chebyshev coefficients to values at Chebyshev points of the 2nd kind.
    
        #   V = COEFFS2VALS(C) returns the values of the polynomial V(i,1) = P(x_i) =
        #   C(1,1)*T_{0}(x_i) + ... + C(N,1)*T_{N-1}(x_i), where the x_i are
        #   2nd-kind Chebyshev nodes.
        #
        #  Input: coeffs is numpy.ndarray 
        #  Output: values is a numpy.ndarray of the same length as coeffs
        #   
        #
        # See also VALS2COEFFS, CHEBPTS.
    
        # Copyright 2016 by The University of Oxford and The Chebfun Developers. 
        # See http://www.chebfun.org/ for Chebfun information.
    
        ################################################################################
        # [Developer Note]: This is equivalent to Discrete Cosine Transform of Type I.
        #
        # [Mathematical reference]: Sections 4.7 and 6.3 Mason & Handscomb, "Chebyshev
        # Polynomials". Chapman & Hall/CRC (2003).
        ################################################################################
        """
    
        # *Note about symmetries* The code below takes steps to 
        # ensure that the following symmetries are enforced:
        # even Chebyshev COEFFS exactly zero ==> VALUES are exactly odd
        # odd Chebychev COEFFS exactly zero ==> VALUES are exactly even
        # These corrections are required because the MATLAB FFT does not
        # guarantee that these symmetries are enforced.
    
        # Make sure everything is in floating point:
        coeffs = 1.0 * coeffs

        # Get the length of the input:
        n = len(coeffs)
    
        # Trivial case (constant or empty):
        if n <= 1:
            values = np.copy(coeffs)
            return values
    
    
        # check for symmetry
        iseven = np.max(np.abs(coeffs[1::2])) == 0.0
        isodd = np.max(np.abs(coeffs[::2])) == 0.0

        # Scale the interior coefficients by 1/2:
        temp = np.copy(coeffs)
        temp[1:n-1] = temp[1:n-1]/2.0
    
        # Mirror the coefficients (to fake a DCT using an FFT):
        tmp = np.r_[temp , temp[n-2:0:-1]]
    
        if np.isreal(coeffs).all():
            # Real-valued case:
            values = np.fft.fft(tmp).real
        elif np.isreal(1.0j*coeffs).all():
            # Imaginary-valued case:
            values = 1.0j*(np.fft.fft(tmp.imag).real)
        else:
            # General case:
            values = np.fft.fft(tmp)
    
        # Flip and truncate:
        values = values[n-1::-1]
    

        # [TODO] Is the np.fft already symmetric? in which 
        # case we don't need this extra enforcing
        if iseven:
            values = (values + np.flipud(values))/2.0
        if isodd:
            values = (values - np.flipud(values))/2.0

        return (1.0 + 0.0j) * values

    @staticmethod
    def clenshaw(x, c):
        """Clenshaw's algorithm for evaluating a Chebyshev expansion with coeffs c at x."""
    
        # Clenshaw scheme for scalar-valued functions.
        bk1 = 0*x
        bk2 = bk1
        x = 2*x
        n = len(c)-1
        # for k = (n+1):-2:3
        for k in np.r_[n:1:-2]:
            # bk2 = c(k) + x.*bk1 - bk2;
            bk2 = c[k] + x*bk1 - bk2;
            # bk1 = c(k-1) + x.*bk2 - bk1;
            bk1 = c[k-1] + x*bk2 - bk1;
    
        if np.mod(n, 2):
            # [bk1, bk2] = deal(c(2) + x.*bk1 - bk2, bk1);
            tmp = bk1
            bk1 = c[1] + x * bk1 - bk2
            bk2 = tmp
    
        y = c[0] + 0.5 * x * bk1 - bk2
        return y

    @staticmethod
    def alias(coeffs, m):
        """Alias Chebyshev coefficients on the 2nd kind Chebyshev grid.

        ALIAS(C, M) aliases the Chebyshev coefficients stored in the column vector C
        to have length M. If M > LENGTH(C), the coefficients are padded with zeros.
        If C is a matrix of coefficients, each of the columns is aliased to length
        M.
        
        # See also PROLONG.

        # Copyright 2016 by The University of Oxford and The Chebfun Developers.
        # See http://www.chebfun.org/ for Chebfun information.

        ################################################################################
        # Useful References:
        #
        #   L.N. Trefethen, Approximation Theory and Approximation Practice, SIAM, 2013
        #   Page 27.
        #
        #   Fox, L. and Parker, I. B., Chebyshev polynomials in Numerical Analysis,
        #   Oxford University Press, 1972.  (pp. 67)
        #
        #   Mason, J. C. and Handscomb, D. C., Chebyshev polynomials, Chapman &
        #   Hall/CRC, Boca Raton, FL, 2003.  (pp. 153)
        #
        ################################################################################
        """
        # Make sure everything is floating point:
        coeffs = 1.0 * coeffs

        n = len(coeffs)

        # Pad with zeros:
        if m > n:
            aliased_coeffs = np.r_[coeffs, Chebtech.zeros(m-n)]
            return aliased_coeffs

        # Alias coefficients: (see eq. (4.4) of Trefethen, Approximation Theory and
        # Approximation Practice, SIAM, 2013):

        aliased_coeffs = Chebtech.zeros(m)

        if m == 0:
            return aliased_coeffs

        if m == 1:
            # Reduce to a single point:
            e = Chebtech.ones(n//2 + n%2)
            e[1::2] = -1.0
            aliased_coeffs = np.dot(e, coeffs[::2])
            return aliased_coeffs


        aliased_coeffs = np.copy(coeffs)
        if m > n/2:
            # If m > n/2, only single coefficients are aliased, and we can vectorise.
            j = np.r_[m:n]
            k = np.abs((j + m - 2)%(2*m-2) - m + 2)
            aliased_coeffs[k] = aliased_coeffs[k] + aliased_coeffs[j];
        else:
            # Otherwise we must do everything in a tight loop. (Which is slower!)
            for j in np.r_[m:n]:
                k = np.abs((j + m - 2)%(2*m-2) - m + 2)
                aliased_coeffs[k] = aliased_coeffs[k] + aliased_coeffs[j]

        # Truncate:
        aliased_coeffs = aliased_coeffs[:m]
        
        return aliased_coeffs

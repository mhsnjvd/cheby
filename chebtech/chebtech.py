import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import copy
import types

def iszero_numerically(v, tol=None):
    """
    Tests if the array v is numerically zero up to 
    the provided tolerance in infinity norm.
    Chebtech default tol is used if no tolerance is provided
    """

    if tol is None:
        tol = Chebtech.__default_tol__

    if np.abs(v).max() < tol:
        return True
    else:
        return False


class Chebtech:
    # Initialize properties of the object
    __default_dtype__ = np.complex128
    __default_tol__ = np.spacing(1)
    __default_min_samples__ = 2**4 + 1
    __default_max_samples__ = 2**16 + 1
    
    def __init__(self, fun=None, **kwargs):
        self.coeffs = np.array([], dtype=type(self).__default_dtype__)
        self.ishappy = False

        keys = kwargs.keys()
        # [TODO]if 'coeffs' in keys and 'values' in keys:
            #except 
        if 'coeffs' in keys:
            coeffs = np.asarray(kwargs['coeffs'], dtype=type(self).__default_dtype__)
            self.coeffs = coeffs.copy()
            self.ishappy = True
        if 'values' in keys:
            values = np.asarray(kwargs['values'], dtype=type(self).__default_dtype__)
            self.coeffs = Chebtech.vals2coeffs(values)
            self.ishappy = True
        if 'coeffs' in keys and 'values' in keys:
            print('values and coeffs both passed, coeffs have been recomputed using values')
        if fun is not None:
            self.ishappy = False
            self.__ctor__(fun)

    def __ctor__(self, op):
        """Adaptive construction of a chebtech object from lambda op"""

        if isinstance(op, str):
            op = eval('lambda x: ' + op)

        #%%%%%%%%%%%%%%%%%%%%%%%%%%% Adaptive construction. %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Initialise empty values to pass to refine:
        values = None
        ishappy = False
        vscale = 0.0
        hscale = 1.0

        # Loop until ISHAPPY or GIVEUP:
        while True:
            # Call the appropriate refinement routine: (in PREF.REFINEMENTFUNCTION)
            values, give_up = Chebtech.refine_by_nesting(op, values)

            # We're giving up! :(
            if give_up:
                self.ishappy = False
                print('Function did not converge after %d samples', len(values))
                return
            
            # Update vertical scale: (Only include sampled finite values)
            values_temp = np.copy(values)
            values_temp[~np.isfinite(values)] = 0.0
            vscale = np.max(np.r_[vscale, np.max(np.abs(values_temp))])

            # Compute the Chebyshev coefficients:
            self.coeffs = Chebtech.vals2coeffs(values)
            
            # Check for happiness:
            ishappy, cutoff = self.standard_check(values, vscale, hscale, Chebtech.__default_tol__)

            # We're happy! :)
            if ishappy:
                # discard unwanted coefficients
                self.coeffs = self.prolong(cutoff)
                self.ishappy = ishappy
                return
            
        ############### Assign to CHEBTECH object. ##############

    def prolong(self, nOut):
        """Manually adjust the number of points used in a CHEBTECH.
        #   C = PROLONG(F, N) returns coeffs C where LENGTH(C) = N and C represents
        #   the same function as F but using more or less coefficients than F.
        #
        #   If N < LENGTH(F) the representation is compressed by chopping
        #   coefficients, which may result in a loss of accuracy.
        #
        #   If N > LENGTH(F) the coefficients are padded with zeros.
        #
        # See also CHEBTECH1/ALIAS, CHEBTECH2/ALIAS.

        # Copyright 2016 by The University of Oxford and The Chebfun Developers.
        # See http://www.chebfun.org/ for Chebfun information.
        """

        # Store the number of values the input function has:
        coeffs = np.copy(self.coeffs)
        nIn = len(coeffs)

        # nDiff is the number of new values needed (negative if compressing).
        nDiff = nOut - nIn

        # Trivial case
        if nDiff == 0:
            # Nothing to do here!
            return coeffs

        # nDiff > 0
        if nDiff > 0:
            coeffs = np.r_[coeffs, Chebtech.zeros(nDiff)]
            return coeffs

        # nDiff < 0
        if nDiff < 0:
            m = np.max(np.r_[nOut, 0])
            coeffs = coeffs[:m]
            return coeffs


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
        y = self(x)
        if not np.all(np.isreal(y)):
            print('Discarding imaginary values in plot')
        y = y.real
        plt.plot(x, y)
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
        """Absolute value of a CHEBTECH object.
         ABS(F) returns the absolute value of F, where F is a CHEBTECH 
         object with no roots in [-1 1]. 
         If ~isempty(roots(F)), then ABS(F) will return garbage
         with no warning. F may be complex.

        Copyright 2016 by The University of Oxford and The Chebfun Developers.
        See http://www.chebfun.org/ for Chebfun information.
        """

        if self.isreal() or self.isimag():
            # Convert to values and then compute ABS(). 
            return Chebtech(values=np.abs(self.values()))
        else:
            # [TODO]
            # f = compose(f, @abs, [], [], varargin{:});
            return Chebtech(lambda x: np.abs(self(x)))

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
            big_coeffs_mask = np.where(np.abs(c) > tail_max)[0]
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

        # Initialise storage:
        # Obtain Chebyshev coefficients {c_r}
        c = self.coeffs.copy()

        n = len(c)

        if n == 0:
            return Chebtech()
        
        # Pad with zeros
        c = np.r_[c, Chebtech.zeros(2)]
        # Initialize vector b = {b_r}
        b = Chebtech.zeros(n+1)
        
        # Compute b_(2) ... b_(n+1):
        b[2:] = (c[1:n] - c[3:n+2])/(2*np.r_[2:n+1])
        # Compute b_1
        b[1] = c[0] - c[2]/2
        v = Chebtech.ones(n)
        v[1::2] = -1.0
        # Compute b_0 (satisfies f(-1) = 0)
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

    def simplify(self, tol=None):
        """Remove small trailing Chebyshev coeffs of a happy CHEBTECH object.
        #  G = SIMPLIFY(F) attempts to compute a 'simplified' version G of the happy
        #  CHEBTECH object F such that LENGTH(G) <= LENGTH(F) but ||G - F|| is small in
        #  a relative sense. It does this by calling the routine STANDARDCHOP.
        #
        #  If F is not happy, F is returned unchanged.
        #
        #  G = SIMPLIFY(F, TOL) does the same as above but uses TOL instead of EPS.  If
        #  TOL is a row vector with as many columns as F, then TOL(k) will be used as
        #  the simplification tolerance for column k of F.
        #
        # See also STANDARDCHOP.

        # Copyright 2016 by The University of Oxford and The Chebfun Developers.
        # See http://www.chebfun.org/ for Chebfun information.
        """

        coeffs = np.copy(self.coeffs)
        # Deal with empty case.
        if len(coeffs) == 0:
            return copy.deepcopy(self)

        # Do nothing to an unhappy CHEBTECH.
        if not self.ishappy:
            return copy.deepcopy(self)

        # STANDARDCHOP requires at least 17 coefficients to avoid outright rejection.
        # STANDARDCHOP also employs a look ahead feature for detecting plateaus. For F
        # with insufficient length the coefficients are padded using prolong. The
        # following parameters are chosen explicitly to work with STANDARDCHOP. 
        # See STANDARDCHOP for details.
        n_old = len(coeffs)
        N = int(np.max(np.r_[Chebtech.__default_min_samples__, np.round(n_old*1.25 + 5)]))
        coeffs = self.prolong(N)

        # After the coefficients of F have been padded with zeros an artificial plateau
        # is created using the noisy output from the FFT. The slightly noisy plateau is
        # required since STANDARDCHOP uses logarithms to detect plateaus and this has
        # undesirable effects when the plateau is made up of all zeros.
        coeffs = Chebtech.vals2coeffs(Chebtech.coeffs2vals(coeffs))

        # Use the default tolerance if none was supplied.
        if tol is None:
            tol = Chebtech.__default_tol__


        cutoff = Chebtech.standard_chop(coeffs, tol)

        # Take the minimum of CUTOFF and LENGTH(F). This is necessary when padding was
        # required.
        cutoff = np.min(np.r_[cutoff, n_old])

        # Chop coefficients using CUTOFF.
        return Chebtech(coeffs=coeffs[:cutoff])

    def flipud(self):
        """FLIPUD   Flip/reverse a CHEBTECH object.
           G = FLIPUD(F) returns G such that G(x) = F(-x) for all x in [-1,1].
        """

        # Negate the odd coefficients:
        self.coeffs[1::2] = -self.coeffs[1::2]
        # Note: the first odd coefficient is at index 2.

    def __pos__(self):
        return self

    def __neg__(self):
        result = copy.deepcopy(self)
        result.coeffs = -result.coeffs
        return result

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

            # N = m + n + 1
            N = len(fc)
            # Toeplitz vector.
            t = np.r_[2.0*fc[0], fc[1:]]
            # Embed in Circulant.
            x = np.r_[2.0*gc[0], gc[1:]]
            # FFT for Circulant mult.
            xprime = np.fft.fft(np.r_[x, x[-1:0:-1]])
            aprime = np.fft.fft(np.r_[t, t[-1:0:-1]])
            # Diag in function space.
            Tfq = np.fft.ifft(aprime * xprime)
            out_coeffs = 0.25 * np.r_[Tfq[0], Tfq[1:] + Tfq[-1:0:-1]]
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
        if isinstance(x, types.LambdaType):
            # A lambda has been passed
            f = x
            # The square brackets are crucial here
            # to avoid infinite recursion See __getitem__
            return Chebtech(lambda x: self[f(x)])
        elif isinstance(x, Chebtech):
            # A Chebtech has been passed
            f = x
            # Use square brackets in both cases
            return Chebtech(lambda x: self[f[x]])
        else:
            # The square brackets are crucial here
            # to avoid infinite recursion See __getitem__
            return self[x]

    def __getitem__(self, x):
        # Evaluate the object at the point(s) x:
        # fx = bary.bary(x, self.values)
        fx = Chebtech.clenshaw(x, self.coeffs)
        return fx

    @staticmethod
    def empty_array():
        return np.array([], dtype=Chebtech.__default_dtype__)

    @staticmethod
    def zeros(n):
        return np.zeros(n, dtype=Chebtech.__default_dtype__)

    @staticmethod
    def ones(n):
        return np.ones(n, dtype=Chebtech.__default_dtype__)

    @staticmethod
    def eye(*args):
        return np.eye(*args, dtype=Chebtech.__default_dtype__)

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
        """Clenshaw's algorithm for evaluating a Chebyshev expansion with coeffs c at x.
        c is assumed to be an array
        x can be a scalar or an array
        y is returned which is scalar or array depending on x
        """


        # Make sure x is of the right data type:
        x = np.array(x, dtype=Chebtech.__default_dtype__)
    
        # Clenshaw scheme for scalar-valued functions.
        bk1 = 0.0*x
        bk2 = np.copy(bk1)
        x = 2.0*x
        n = len(c)
        for k in np.r_[n-1:1:-2]:
            bk2 = c[k] + x*bk1 - bk2
            bk1 = c[k-1] + x*bk2 - bk1
    
        if not np.mod(n, 2):
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


    @staticmethod
    def standard_chop(coeffs, tol=None):
        """A sequence chopping rule of "standard" (as opposed to "loose" or
        # "strict") type, that is, with an input tolerance TOL that is applied with some
        # flexibility.  This code is used in all parts of Chebfun that make chopping
        # decisions, including chebfun construction (CHEBTECH, TRIGTECH), solution of
        # ODE BVPs (SOLVEBVP), solution of ODE IVPs (ODESOL), simplification of chebfuns
        # (SIMPLIFY), and Chebfun2.  See J. L. Aurentz and L. N. Trefethen, "Chopping a
        # Chebyshev series", http://arxiv.org/abs/1512.01803, December 2015.
        #
        # Input:
        #
        # COEFFS  A nonempty row or column vector of real or complex numbers
        #         which typically will be Chebyshev or Fourier coefficients.
        #
        # TOL     A number in (0,1) representing a target relative accuracy.
        #         TOL will typically will be set to the Chebfun EPS parameter,
        #         sometimes multiplied by a factor such as vglobal/vlocal in
        #         construction of local pieces of global chebfuns.
        #         Default value: machine epsilon (MATLAB EPS).
        #
        # Output:
        #
        # CUTOFF  A positive integer.
        #         If CUTOFF == length(COEFFS), then we are "not happy":
        #         a satisfactory chopping point has not been found.
        #         If CUTOFF < length(COEFFS), we are "happy" and CUTOFF
        #         represents the last index of COEFFS that should be retained.
        #
        # Examples:
        #
        # coeffs = 10.^-(1:50); random = cos((1:50).^2);
        # standardChop(coeffs) % = 18
        # standardChop(coeffs + 1e-16*random) % = 15
        # standardChop(coeffs + 1e-13*random) % = 13
        # standardChop(coeffs + 1e-10*random) % = 50
        # standardChop(coeffs + 1e-10*random, 1e-10) % = 10
         
        # Jared Aurentz and Nick Trefethen, July 2015.
        #
        # Copyright 2016 by The University of Oxford and The Chebfun Developers. 
        # See http://www.chebfun.org/ for Chebfun information.

        # STANDARDCHOP normally chops COEFFS at a point beyond which it is smaller than
        # TOL^(2/3).  COEFFS will never be chopped unless it is of length at least 17 and
        # falls at least below TOL^(1/3).  It will always be chopped if it has a long
        # enough final segment below TOL, and the final entry COEFFS(CUTOFF) will never
        # be smaller than TOL^(7/6).  All these statements are relative to
        # MAX(ABS(COEFFS)) and assume CUTOFF > 1.  These parameters result from
        # extensive experimentation involving functions such as those presented in
        # the paper cited above.  They are not derived from first principles and
        # there is no claim that they are optimal.
        """

        # Set default if fewer than 2 inputs are supplied: 
        # [TODO]: How to set tolerance to some default:
        if tol is None:
            tol = Chebtech.__default_tol__

        # Check magnitude of TOL:
        if tol >= 1:
            cutoff = 1
            return cutoff

        # Make sure COEFFS has length at least 17:
        n = len(coeffs)
        cutoff = n
        if n < 17:
            return cutoff
          
        # Step 1: Convert COEFFS to a new monotonically nonincreasing
        #         vector ENVELOPE normalized to begin with the value 1.

        b = np.abs(coeffs)
        m = b[-1]*Chebtech.ones(n)
        for j in np.r_[n-2:-1:-1]:
            m[j] = np.max(np.r_[b[j], m[j+1]])

        if m[0] == 0.0:
            cutoff = 1
            return cutoff

        envelope = m/m[0]

        # For Matlab version 2014b and later step 1 can be computed using the
        # cummax command.
        # envelope = cummax(abs(coeffs),'reverse');
        # if envelope(1) == 0
        #     cutoff = 1;
        #     return
        # else
        #     envelope = envelope/envelope(1);
        # end

        # Step 2: Scan ENVELOPE for a value PLATEAUPOINT, the first point J-1, if any,
        # that is followed by a plateau.  A plateau is a stretch of coefficients
        # ENVELOPE(J),...,ENVELOPE(J2), J2 = round(1.25*J+5) <= N, with the property
        # that ENVELOPE(J2)/ENVELOPE(J) > R.  The number R ranges from R = 0 if
        # ENVELOPE(J) = TOL up to R = 1 if ENVELOPE(J) = TOL^(2/3).  Thus a potential
        # plateau whose starting value is ENVELOPE(J) ~ TOL^(2/3) has to be perfectly
        # flat to count, whereas with ENVELOPE(J) ~ TOL it doesn't have to be flat at
        # all.  If a plateau point is found, then we know we are going to chop the
        # vector, but the precise chopping point CUTOFF still remains to be determined
        # in Step 3.

        for j in np.r_[2:n+1]:
            j2 = int(np.round(1.25*j + 5))
            if j2 > n:
                # there is no plateau: exit
                return cutoff

            e1 = envelope[j-1]
            e2 = envelope[j2-1]
            if e1 == 0.0:
                plateau_point = j - 1
                break
            elif (e2/e1) > (3.0*(1.0 - np.log(e1)/np.log(tol))):
                # a plateau has been found: go to Step 3
                plateau_point = j - 1
                break

        # Step 3: fix CUTOFF at a point where ENVELOPE, plus a linear function
        # included to bias the result towards the left end, is minimal.
        #
        # Some explanation is needed here.  One might imagine that if a plateau is
        # found, then one should simply set CUTOFF = PLATEAUPOINT and be done, without
        # the need for a Step 3. However, sometimes CUTOFF should be smaller or larger
        # than PLATEAUPOINT, and that is what Step 3 achieves.
        #
        # CUTOFF should be smaller than PLATEAUPOINT if the last few coefficients made
        # negligible improvement but just managed to bring the vector ENVELOPE below the
        # level TOL^(2/3), above which no plateau will ever be detected.  This part of
        # the code is important for avoiding situations where a coefficient vector is
        # chopped at a point that looks "obviously wrong" with PLOTCOEFFS.
        #
        # CUTOFF should be larger than PLATEAUPOINT if, although a plateau has been
        # found, one can nevertheless reduce the amplitude of the coefficients a good
        # deal further by taking more of them.  This will happen most often when a
        # plateau is detected at an amplitude close to TOL, because in this case, the
        # "plateau" need not be very flat.  This part of the code is important to
        # getting an extra digit or two beyond the minimal prescribed accuracy when it
        # is easy to do so.

        if envelope[plateau_point-1] == 0.0:
            cutoff = plateau_point
            return cutoff
        else:
            j3 = np.count_nonzero(envelope >= tol**(7.0/6.0))
            if j3 < j2:
                j2 = j3 + 1
                envelope[j2-1] = tol**(7.0/6.0)

            cc = np.log10(envelope[:j2])
            cc = cc + np.linspace(0, (-1.0/3.0)*np.log10(tol), j2)
            d = np.argmin(cc)
            cutoff = np.max(np.r_[d, 1])
            return cutoff


    def standard_check(self, values=None, vscale=0.0, hscale=1.0, tol=None):
        """Attempt to trim trailing Chebyshev coefficients in a CHEBTECH.
        #   [ISHAPPY, CUTOFF] = STANDARDCHECK(F) uses the routine STANDARDCHOP to
        #   compute a positive integer CUTOFF which represents the number of
        #   coefficients of F that are deemed accurate enough to keep. ISHAPPY is TRUE
        #   if the CUTOFF value returned by STANDARDCHOP is less than LENGTH(F) and
        #   FALSE otherwise.
        #
        #   [ISHAPPY, CUTOFF] = STANDARDCHECK(F, VALUES, DATA, PREF) allows additional
        #   preferences to be passed. VALUES is a matrix of the function values of F at
        #   the corresponding interpolation points. DATA.VSCALE is an approximation of
        #   the maximum function value of F on a possibly larger approximation
        #   interval.  PREF is a data structure used to pass in additional information,
        #   e.g. a target accuracy tolerance could be passed using PREF.CHEBFUNEPS.
        #
        # See also CLASSICCHECK, STRICTCHECK, LOOSECHECK.

        # Copyright 2016 by The University of Oxford and The Chebfun Developers.
        # See http://www.chebfun.org/ for Chebfun information.
        """

        # Grab the coefficients
        coeffs = np.copy(self.coeffs)
        n = len(coeffs)

        # Check for NaNs and exit if any are found.
        if np.any(np.isnan(coeffs)):
            print('Chebtech:standard_check: nan encountered in coeffs')

        #[TODO]: handle tolerance:
        if tol is None:
            tol = Chebtech.__default_tol__

        # Compute function values of F if none were given.
        if values is None:
            values = Chebtech.coeffs2vals(coeffs)

        # Scale TOL by the MAX(DATA.HSCALE, DATA.VSCALE/VSCALE(F)).
        # This choice of scaling is the result of undesirable behavior when using
        # standardCheck to construct the function f(x) = sqrt(1-x) on the interval [0,1]
        # with splitting turned on. Due to the way standardChop checks for plateaus, the
        # approximations on the subdomains were chopped incorrectly leading to poor
        # quality results. This choice of scaling corrects this by giving less weight to
        # subintervals that are much smaller than the global approximation domain, i.e.
        # HSCALE >> 1. For functions on a single domain with no breaks, this scaling has
        # no effect since HSCALE = 1.
        vscaleF = np.max(np.abs(values))

        #Avoid divide by zero if all values are zero
        if vscaleF == 0.0:
            vscaleF = 1.0

        tol = tol * np.max(np.r_[hscale, vscale/vscaleF])

        # Chop the coefficients:
        cutoff = Chebtech.standard_chop(coeffs[:], tol)

        # Check for happiness.
        ishappy = (cutoff < n)
        
        return ishappy, cutoff

    @staticmethod
    def refine_by_resampling(op, values):
        """REFINERESAMPLING   Default refinement function for resampling scheme.
        """

        min_samples = Chebtech.__default_min_samples__
        max_samples = Chebtech.__default_max_samples__

        if (values is None) or (len(values) == 0):
            # Choose initial n based upon min_samples:
            n = int(2.0 ** np.ceil(np.log2(min_samples - 1)) + 1)
        else:
            # (Approximately) powers of sqrt(2):
            power = np.log2(len(values) - 1)
            if (power == np.floor(power)) and (power > 5):
                n = int(np.round(2.0**(np.floor(power) + 0.5)) + 1)
                n = n - mod(n, 2) + 1
            else:
                n = int(2.0**(np.floor(power) + 1) + 1)
        
        # n is too large:
        if n > max_samples:
            # Don't give up if we haven't sampled at least once.
            if len(values) == 0:
                n = max_samples
                give_up = False
            else:
                give_up = True
                return values, give_up
        else:
            give_up = False
       
        # 2nd-kind Chebyshev grid:
        x = Chebtech.chebpts(n)

        values = op(x)
        
        return values, give_up

    @staticmethod
    def refine_by_nesting(op, values):
        """REFINENESTED  Default refinement function for single ('nested') sampling."""

        min_samples = Chebtech.__default_min_samples__
        max_samples = Chebtech.__default_max_samples__

        if (values is None) or (len(values) == 0):
            # The first time we are called, there are no values
            # and REFINENESTED is the same as REFINERESAMPLING.
            values, give_up = Chebtech.refine_by_resampling(op, values)
            return values, give_up
        else:
            # Compute new n by doubling (we must do this when not resampling).
            n = 2*len(values) - 1
            
            # n is too large:
            if n > max_samples:
                give_up = True
                return values, give_up
            else:
                give_up = False
            
            # 2nd-kind Chebyshev grid:
            x = Chebtech.chebpts(n);
            # Take every 2nd entry:
            # x = x(2:2:end-1);
            x = x[1:-1:2]

            # Insert the old values:
            new_values = Chebtech.zeros(n)
            new_values[:n:2] = values
            # Compute and insert new ones:
            new_values[1::2] = op(x)
            return new_values, give_up

# @staticmethod
# def classic_check(coeffs, values=None, data=None):
#     """CLASSICCHECK   Attempt to trim trailing Chebyshev coefficients in a CHEBTECH.
#     [ISHAPPY, CUTOFF] = CLASSICCHECK(F, VALUES, DATA) returns an estimated
#     location, the CUTOFF, at which the CHEBTECH F could be truncated to
#     maintain an accuracy of EPSLEVEL (see documentation below) relative to
#     DATA.VSCALE and DATA.HSCALE. ISHAPPY is TRUE if the representation is
#     "happy" in the sense described further below and FALSE otherwise.
#     
#     [ISHAPPY, CUTOFF] = CLASSICCHECK(F, VALUES, DATA, PREF) allows additional
#     preferences to be passed. In particular, one can adjust the target accuracy
#     with PREF.CHEBFUNEPS.
#     
#     CLASSICCHECK first queries HAPPINESSREQUIREMENTS to obtain TESTLENGTH and
#     EPSLEVEL (see documentation below). If |F.COEFFS(1:TESTLENGTH)|/VSCALE <
#     EPSLEVEL, then the representation defined by F.COEFFS is deemed happy. The
#     value returned in CUTOFF is essentially that from TESTLENGTH (although it
#     can be reduced if there are further COEFFS which fall below EPSLEVEL).
#     
#     HAPPINESSREQUIREMENTS defines what it means for a CHEBTECH to be happy.
#     [TESTLENGTH, EPSLEVEL] = HAPPINESSREQUIREMENTS(VALUES, COEFFS, POINTS,
#     DATA, EPS) returns two scalars TESTLENGTH and EPSLEVEL.  POINTS 
#     is the vector of points at which F was sampled to get the values in 
#     VALUES.  EPS is the desired accuracy.  A CHEBTECH is deemed to be 
#     'happy' if the coefficients COEFFS(END-TESTLENGTH+1:END) (recall that 
#     COEFFS are stored in ascending order) are all below EPSLEVEL.  The 
#     default choice of the test length is:
#         TESTLENGTH = n,             for n = 1:4
#         TESTLENGTH = 5,             for n = 5:44
#         TESTLENGTH = round((n-1)/8) for n > 44
#     
#     EPSLEVEL is essentially the maximum of:
#         * pref.chebfuneps
#         * eps*TESTLENGTH
#         * eps*condEst (where condEst is an estimate of the condition number
#                        based upon a finite difference approximation to the
#                        gradient of the function from VALUES.).
#     However, the final two estimated values can be no larger than 1e-4.
#     
#     Note that the accuracy check implemented in this function is (roughly) the
#     same as that employed in Chebfun v4.x.
#     See also STRICTCHECK, LOOSECHECK.
#     """
# 
# 
#     def happinessRequirements(values, coeffs, x, vscale, hscale, eps_level):
#         """Define what it means for a CHEBTECH to be happy.
#         """
#         #   See documentation above.
# 
#         # Grab the size:
#         n = len(values)
# 
#         # We will not allow the estimated rounding errors to be cruder than this value:
#         # Worst case precision!
#         min_prec = 1.0e-4; 
# 
#         # Length of tail to test.
#         n_1 = np.max(np.r_[5.0, np.round((1.0*(n-1))/8.0)])
#         test_length = np.min(np.r_[n, n_1])
# 
#         # Look at length of tail to loosen tolerance:
#         err_1 = np.spacing(1)*test_length
#         tail_err = np.min(np.r_[err_1, min_prec])
# 
#         # Estimate the condition number of the input function by
#         #    ||f(x+eps(x)) - f(x)||_inf / ||f||_inf ~~ (eps(hscale)/vscale)*f'.
#         dy = np.diff(values)
#         dx = np.diff(x)
#         grad_est = np.max(np.abs(dy/dx))                        # Finite difference approx
#         cond_est = np.spacing(hscale)/vscale*grad_est # Condition number estimate.
#         cond_est = np.min(np.r_[cond_est, min_prec])
# 
#         # Choose maximum between prescribed tolerance and estimated rounding errors:
#         eps_level_1 = np.max(np.r_[eps_level, cond_est])
#         eps_level = np.max(np.r_[eps_level_1, tail_err])
# 
#         return test_length, eps_level
# 
# 
#     # Determine n (the length of the input).
#     n = f.length()
# 
#     # Assume we're not happy. (N'aww! :( )
#     ishappy = False
# 
#     # Deal with the trivial case:
#     if n < 2: 
#         # (Can't be simpler than a constant!)
#         cutoff = n
#         return cutoff
# 
#     # NaNs are not allowed.
#     if np.any(np.isnan(f.coeffs)):
#         print('CHEBFUN:CHEBTECH:classicCheck:nanEval')
#         return
#     # Compute some values if none were given:
#     if values is None:
#         values = f.coeffs2vals(f.coeffs)
# 
#     # Check the vertical scale:
#     if vscale == 0.0:
#         # This is the zero function, so we must be happy!
#         ishappy = True
#         cutoff = 1
#         return cutoff
#     elif np.isinf(vscale):
#         # Inf located. No cutoff.
#         cutoff = n
#         return cutoff
#     else:
#         # We need this for constructing the zero function:
#         pass
#         #data.vscale(~data.vscale) = 1;
# 
#     # Check for convergence and chop location --------------------------------------
# 
#     # Absolute value of coefficients, relative to vscale:
#     ac = np.abs(f.coeffs)/vscale
# 
#     # Happiness requirements:
#     test_length, eps_level = happinessRequirements(values, f.coeffs, f.points(), vscal, hscale, eps_level)
#     # We have converged! Chop tail:
#     if np.all(np.max(ac[-test_length:]) < eps_level):
#         # We must be happy.
#         ishappy = True
#         # Find last row of coeffs with entry above epslevel:
#         large_coeffs_loc = np.where(ac > eps_level)[0]
#         if len(large_coeffs_loc) == 0:
#             tail_loc = None
#         else:
#             tail_loc = np.max(large_coeffs_loc) + 1
# 
#         # Check for the zero function!
#         if tail_loc is None:
#             cutoff = 1
#             return ishappy, cutoff
# 
#         # Compute the cumulative max of eps/4 and the tail entries:
#         t = .25*np.spacing(1)
#         # Restrict to coefficients of interest.
#         ac = np.flipud(ac[tail_loc:])
#         # Cumulative maximum.
#         #for k = 1:size(ac, 1)           
#         for k in range(len(ac)):
#             pass
#             #ind = ac(k, :) < t;
#             #ac(k, ind) = t(ind);
#             #ind = ac(k, :) >= t;
#             #t(ind) = ac(k, ind);
# 
#         # Obtain an estimate for how much accuracy we'd gain compared to reducing
#         # length ("bang for buck"):
#         bang = np.log(1.0e3*eps_level/ac)
#         buck = 1.0*np.r_[n-1:tail_loc:-1]
#         Tbpb = bang / buck
# 
#         # Compute position at which to chop.  Keep greatest number of coefficients
#         # demanded by any of the columns.
#         #[ignored, perColTchop] = max(Tbpb(3:n-Tloc+1, :));
#         #Tchop = min(perColTchop);
# 
#         # We want to keep [c(0), c(1), ..., c(cutoff)]:
#         cutoff = n - Tchop - 2;
# 
#     else:
# 
#         # We are unhappy. :(
#         cutoff = 0
#         
#         # Estimate the epslevel:
#         eps_level = np.mean(ac[-test_length:])
# 
# 

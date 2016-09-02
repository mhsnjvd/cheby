import numpy as np
from scipy import linalg
#import bary
import matplotlib.pyplot as plt

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
    default_dtype = np.float64
    
    def __init__(self, fun=None, **kwargs):
        self.coeffs = np.array([], dtype=type(self).default_dtype)
        self.values = np.array([], dtype=type(self).default_dtype)

        keys = kwargs.keys()
        # [TODO]if 'coeffs' in keys and 'values' in keys:
            #except 
        if 'coeffs' in keys:
            coeffs = np.asarray(kwargs['coeffs'], dtype=type(self).default_dtype)
            self.coeffs = coeffs.copy()
            self.values = Chebtech.coeffs2vals(coeffs)
            self.ishappy = True
        if 'values' in keys:
            values = np.asarray(kwargs['values'], dtype=type(self).default_dtype)
            self.values = values.copy()
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
                    self.coeffs = Chebtech.zero_array()
                    self.values = Chebtech.zero_array()
                    return
                else:
                    n = coeffs_mask[-1] + 1
                    self.coeffs = coeffs[:n]
                    # Sample values again 
                    self.values = f(Chebtech.chebpts(len(self.coeffs)))
                    return
            else:
                n = 2*n
        print('Function did not converge')
        self.ishappy = False


    def length(self):
        return len(self.coeffs)

    def vscale(self):
        if self.length() == 0:
            return np.nan
        else:
            return np.abs(self.values).max()

    def __getitem__(self, x):
        # Evaluate the object at the point(s) x:
        # fx = bary.bary(x, self.values)
        fx = Chebtech.clenshaw(x, self.coeffs)
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

    def roots(self, **kwargs):
        """Roots of a CHEBTECH in the interval [-1,1].
        """

        def rootsunit_coeffs(c, htol):
            """Compute the roots of the polynomial given by the coefficients c on the unit interval."""
       
            # # Define these as persistent, need to compute only once.
            # persistent Tleft Tright
       
            # # Simplify the coefficients:
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
                    r = Chebtech.zero_array()
                else:
                    r = Chebtech.empty_array()
            elif n == 1:
                if c[0] == 0.0 and roots_pref['zero_fun']:
                    r = Chebtech.zero_array()
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
            elif roots_pref['recurse'] and n <= max_eig_size:
                c_old = c.copy()
                c = -0.5 * c[:-1]/c[-1]
                c[-2] = c[-2] + 0.5

                oh = 0.5 * np.ones(len(c)-1)
                A = np.diag(oh, 1) + np.diag(oh, -1)
                A[-2, -1] = 1.0
                A[:, 0] = np.flipud(c)

                if roots_pref['qz']: 
                    B = np.eye(A.shape)
                    c_old = c_old / np.abs(c_old).max()
                    B[0, 0] = c_old[-1]
                    c_old = -0.5 * c_old[:-1]
                    c_old[-2] = c_old[-2] + 0.5 * B[0, 0]
                    A[:, 0] = np.flipud[c_old]
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
            elif ( n <= 513 ):
                ## If n <= 513 then we can compute the new coefficients with a
                ## matrix-vector product.
                # assemble the matrices TLEFT and TRIGHT
                # [TODO] how to make Tleft persistent?
                if ( True ):
                    # # Create the coefficients for TLEFT using the FFT directly:
                    x = chebptsAB(513, [-1, split_point]);
                    Tleft = np.ones((513, 513)); 
                    Tleft[:, 1] = x
                    for k in range(2, 513):
                        Tleft[:, k] = 2 * x * Tleft[:, k-1] - Tleft[:, k-2]

                    Tleft = np.r_[Tleft[-1:0:-1, :], Tleft[:-1, :]]
                    Tleft = np.fft.fft(Tleft).real / 512.0
                    Tleft[0, :] = 0.5 * Tleft[0, :]
                    Tleft[-1, :] = 0.5 * Tleft[0, :]
                    Tleft = np.triu(Tleft)

                    # # Create the coefficients for TRIGHT much in the same way:
                    x = chebptsAB(513, [split_point, 1]);
                    Tright = np.ones((513, 513)); 
                    Tright[:, 1] = x;
                    for k in range(2, 513):
                        Tright[:, k] = 2 * x * Tright[:, k-1] - Tright[:, k-2]; 

                    Tright = np.r_[ Tright[-1:0:-1, :], Tright[:-1, :]]
                    Tright = np.fft.fft(Tright).real / 512.0
                    Tright[0, :] = 0.5 * Tright[0, :]
                    Tright[-1, :] = 0.5 * Tright[-1, :]
                    Tright = np.triu(Tright)
       
                # Compute the new coefficients:
                c_left = np.dot(Tleft[:n, :n], c)
                c_right = np.dot(Tright[:n, :n], c)
       
                # Recurse:
                r = np.r_[ (split_point - 1.0)/2.0 + (split_point + 1.0)/2.0*rootsunit_coeffs(c_left, 2*htol), \
                        (split_point + 1.0)/2.0 + (1.0 - split_point)/2.0*rootsunit_coeffs(c_right, 2*htol) ]
       
            # Otherwise, split using more traditional methods (i.e., Clenshaw):
            else:
                # Evaluate the polynomial on both intervals:
                xx = np.r_[chebptsAB(n, [ -1, split_point]), \
                    chebptsAB(n, [split_point, 1])]
                v = Chebtech.clenshaw(xx, c);
       
                # Get the coefficients on the left:
                c_left = Chebtech.vals2coeffs(v[:n])
       
                # Get the coefficients on the right:
                c_right = Chebtech.vals2coeffs(v[n:])
       
                # Recurse:
                r = np.r_[ (split_point - 1.0)/2.0 + (split_point + 1.0)/2.0*rootsunit_coeffs(c_left, 2*htol), \
                        (split_point + 1.0)/2.0 + (1.0 - split_point)/2.0*rootsunit_coeffs(c_right, 2*htol) ]

            return np.r_[r]


                
       
            ## If n <= 513 then we can compute the new coefficients with a
            ## matrix-vector product.
            #elseif ( n <= 513 )
            #    
            #    # Have we assembled the matrices TLEFT and TRIGHT?
            #    if ( isempty(Tleft) )
            #        # Create the coefficients for TLEFT using the FFT directly:
            #        x = chebptsAB(513, [-1, splitPoint]);
            #        Tleft = ones(513); 
            #        Tleft(:,2) = x;
            #        for k = 3:513
            #            Tleft(:,k) = 2 * x .* Tleft(:,k-1) - Tleft(:,k-2); 
            #        end
            #        Tleft = [ Tleft(513:-1:2,:) ; Tleft(1:512,:) ];
            #        Tleft = real(fft(Tleft) / 512);
            #        Tleft = triu( [ 0.5*Tleft(1,:) ; Tleft(2:512,:) ; 0.5*Tleft(513,:) ] );
       
            #        # Create the coefficients for TRIGHT much in the same way:
            #        x = chebptsAB(513, [splitPoint,1]);
            #        Tright = ones(513); 
            #        Tright(:,2) = x;
            #        for k = 3:513
            #            Tright(:,k) = 2 * x .* Tright(:,k-1) - Tright(:,k-2); 
            #        end
            #        Tright = [ Tright(513:-1:2,:) ; Tright(1:512,:) ];
            #        Tright = real(fft(Tright) / 512);
            #        Tright = triu( [ 0.5*Tright(1,:) ; Tright(2:512,:) ; 0.5*Tright(513,:) ] );
            #    end
       
            #    # Compute the new coefficients:
            #    cleft = Tleft(1:n,1:n) * c;
            #    cright = Tright(1:n,1:n) * c;
       
            #    # Recurse:
            #    r = [ (splitPoint - 1)/2 + (splitPoint + 1)/2*rootsunit_coeffs(cleft, 2*htol) ;
            #          (splitPoint + 1)/2 + (1 - splitPoint)/2*rootsunit_coeffs(cright, 2*htol) ];
       
            ## Otherwise, split using more traditional methods (i.e., Clenshaw):
            #else
            #    
            #    # Evaluate the polynomial on both intervals:
            #    v = chebtech.clenshaw([ chebptsAB(n, [ -1, splitPoint ]) ; ...
            #        chebptsAB(n, [ splitPoint, 1 ]) ], c);
       
            #    # Get the coefficients on the left:
            #    cleft = chebtech2.vals2coeffs(v(1:n));            
       
            #    # Get the coefficients on the right:
            #    cright = chebtech2.vals2coeffs(v(n+1:end));           
       
            #    # Recurse:
            #    r = [ (splitPoint - 1)/2 + (splitPoint + 1)/2*rootsunit_coeffs(cleft, 2*htol) ;
            #          (splitPoint + 1)/2 + (1 - splitPoint)/2*rootsunit_coeffs(cright, 2*htol) ];
       
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
        roots_pref['recurse'] = kwargs.setdefault('recurse', True)
        roots_pref['prune'] = kwargs.setdefault('prune', False)
        roots_pref['zero_fun'] = kwargs.setdefault('zero_fun', True)
        roots_pref['qz'] = kwargs.setdefault('qz', False)
        roots_pref['filter'] = kwargs.setdefault('filter', None)

        if 'complex' in kwargs.keys():
            roots_pref['complex'] = True
            roots_pref['all'] = True
            roots_pref['prune'] = True

        # Subdivision maps [-1,1] into [-1, split_point] and [split_point, 1].
        # This is an arbitrary number.
        split_point = -0.004849834917525


        # Trivial case for f constant:
        if self.length() == 1:
            if self.coeffs[0] == 0.0 and roots_pref['zero_fun']:
                # Return a root at centre of domain:
                out = Chebtech.zero_array()
            else:
                # Return empty:
                out = Chebtech.empty_array()
            return out

        # Get scaled coefficients for the recursive call:
        c = self.coeffs.copy()/self.vscale()

        # Call the recursive rootsunit function:
        # TODO:  Does the tolerance need to depend on some notion of hscale?
        default_tol = np.spacing(1)*100
        r = rootsunit_coeffs(c, default_tol)

        # Try to filter out spurious roots:
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
        c = np.r_[c, np.zeros(2)]
        # b = zeros(n+1, m);                 # Initialize vector b = {b_r}
        b = np.zeros(n+1)
        
        # # Compute b_(2) ... b_(n+1):
        # b(3:n+1,:) = (c(2:n,:) - c(4:n+2,:)) ./ repmat(2*(2:n).', 1, m);
        b[2:] = (c[1:n] - c[3:n+2])/(2*np.r_[2:n+1])
        # b(2,:) = c(1,:) - c(3,:)/2;        # Compute b_1
        b[1] = c[0] - c[2]/2
        # v = ones(1, n);
        v = np.ones(n)
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
                return Chebtech.zero_array()


            cout = np.zeros(n-1)
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
            

    def __add__(self, other):
        if not isinstance(other, Chebtech):
            return self.__radd__(other)

        result = Chebtech()
        n = self.length()
        m = other.length()
        if n >= m:
            coeffs = np.r_[other.coeffs, np.zeros(n-m)]
            result.coeffs = self.coeffs + coeffs
        else:
            coeffs = np.r_[self.coeffs, np.zeros(m-n)]
            result.coeffs = other.coeffs + coeffs

        result.values = Chebtech.coeffs2vals(result.coeffs)

        return result


    def __sub__(self, other):
        if not isinstance(other, Chebtech):
            return self.__rsub__(other)

        result = Chebtech()
        n = self.length()
        m = other.length()
        if n >= m:
            coeffs = np.r_[other.coeffs, np.zeros(n-m)]
            result.coeffs = self.coeffs - coeffs
        else:
            coeffs = np.r_[self.coeffs, np.zeros(m-n)]
            result.coeffs = other.coeffs - coeffs

        result.values = Chebtech.coeffs2vals(result.coeffs)

        return result

    def __radd__(self, other):
        result = Chebtech()
        n = self.length()
        if n == 0:
            result = Chebtech()
        else:
            result.values = self.values + other
            result.coeffs = Chebtech.vals2coeffs(result.values)

        return result

    def __rsub__(self, other):
        return self.__radd__(-1.0*other)

    def __mul__(self, other):
        if not isinstance(other, Chebtech):
            return self.__rmul__(other)

        result = Chebtech()
        n = self.length()
        m = other.length()
        if n >= m:
            coeffs = np.r_[other.coeffs, np.zeros(n-m)]
            result.coeffs = self.coeffs - coeffs
        else:
            coeffs = np.r_[self.coeffs, np.zeros(m-n)]
            result.coeffs = other.coeffs - coeffs

        result.values = Chebtech.coeffs2vals(result.coeffs)

        return result

    def __rmul__(self, other):
        result = Chebtech()
        n = self.length()
        if n == 0:
            result = Chebtech()
        else:
            result.values = self.values * other
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


    @staticmethod
    def empty_array():
        return np.array([], dtype=Chebtech.default_dtype)

    @staticmethod
    def zero_array():
        return np.array([0.0], dtype=Chebtech.default_dtype)

    @staticmethod
    def chebpts(n):
        """CHEBPTS   Chebyshev points in [-1, 1]."""

        # Special case (no points)
        if n == 0:
            x = Chebtech.empty_array()
        # Special case (single point)
        elif n == 1:
            x = Chebtech.zero_array()
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
    
        return coeffs
    
    
    
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
        elif np.isreal(1j*coeffs).all():
            # Imaginary-valued case:
            values = 1j*(np.fft.fft(tmp.imag).real)
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

        return values

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
        """
        #ALIAS   Alias Chebyshev coefficients on the 2nd kind Chebyshev grid.
        #   ALIAS(C, M) aliases the Chebyshev coefficients stored in the column vector C
        #   to have length M. If M > LENGTH(C), the coefficients are padded with zeros.
        #   If C is a matrix of coefficients, each of the columns is aliased to length
        #   M.
        #
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
            aliased_coeffs = np.r_[coeffs, np.zeros(m-n)]
            return aliased_coeffs

        # Alias coefficients: (see eq. (4.4) of Trefethen, Approximation Theory and
        # Approximation Practice, SIAM, 2013):

        aliased_coeffs = np.zeros(m)

        if m == 0:
            return aliased_coeffs

        if m == 1:
            # Reduce to a single point:
            e = np.ones(n//2 + n%2)
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

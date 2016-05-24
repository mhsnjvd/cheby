import numpy as np

def roots(f, **kwargs):
    """
    #ROOTS   Roots of a CHEBTECH in the interval [-1,1].
    #   ROOTS(F) returns the real roots of the CHEBTECH F in the interval [-1,1].
    #
    #   ROOTS(F, PROP1, VAL1, PROP2, VAL2, ...) modifies the default ROOTS
    #   properties. The PROPs (strings) and VALs may be any of the following:
    #
    #   ALL: 
    #       [0] - Return only real-valued roots in [-1,1].
    #        1  - Return roots outside of [-1,1] (including complex roots).
    #
    #   COMPLEX:
    #       [0] - No effect.
    #        1  - Equivalent to setting both PRUNE and ALL = 1.
    #
    #   FILTER:
    #       [ ]
    #   @filter(R,F) - A function handle which accepts the sorted computed roots, R, 
    #                  and the CHEBTECH, F, and filters the roots as it see fit.
    #   RECURSE:
    #        0  - Compute roots without interval subdivision (slower).
    #       [1] - Subdivide until length(F) < 50. (Can cause additional complex
    #             roots).
    #
    #   PRUNE:
    #       [0]
    #        1  - Prune 'spurious' complex roots if ALL == 1 and RECURSE == 0.
    #
    #   QZ: 
    #       [0] - Use the colleague matrix linearization and the QR algorithm.
    #        1  - Use the colleague matrix pencil linearization and the QZ 
    #             algorithm for potentially extra numerical stability. 
    #
    #   ZEROFUN:
    #        0  - Return empty if F is identically 0.
    #       [1] - Return a root at x = 0 if F is identically 0.
    #
    #   If F is an array-valued CHEBTECH then there is no reason to expect each
    #   column to have the same number of roots. In order to return a useful output,
    #   the roots of each column are computed and then padded with NaNs so that a
    #   matrix may be returned. The columns of R = ROOTS(F) correspond to the
    #   columns of F.

    # Copyright 2016 by The University of Oxford and The Chebfun Developers. 
    # See http://www.chebfun.org/ for Chebfun information.

    ################################################################################
    # ROOTS works by recursively subdividing the interval until the resulting
    # CHEBTECH is of degree less than 50, at which point a colleague matrix is
    # constructed to compute the roots.
    #
    # ROOTS performs all operations in coefficient space. In this representation,
    # two matrices, TLEFT and TRIGHT (both of size 512 by 512), are constructed such
    # that TLEFT*C and TRIGHT*C are the coefficients of the polynomials in the left
    # and right intervals respectively. This is faster than evaluating the
    # polynomial using the barycentric formula or Clenshaw's algorithm in the
    # respective intervals despite both computations requiring O(N^2) operations.
    #
    # For polynomials of degree larger than 512, the interval is subdivided by
    # evaluating on the left and right intervals using the Clenshaw algorithm. The
    # subdivision occurs at an arbitrary point _near_ but not _at_ the centre of the
    # domain (in fact, -0.004849834917525 on [-1 1]) to avoid introducing additional
    # spurious roots (since x = 0 is often a special point).
    #
    # Note that ROOTS uses CHEBTECH2 technology to subdivide the interval,
    # regardless of whether F is a CHEBTECH1 or a CHEBTECH2.
    #
    # [Mathematical references]:
    #  * I. J. Good, "The colleague matrix, a Chebyshev analogue of the companion
    #    matrix", Quarterly Journal of Mathematics 12 (1961).
    #
    #  * J. A. Boyd, "Computing zeros on a real interval through Chebyshev expansion
    #    and polynomial rootfinding", SIAM Journal on Numerical Analysis 40 (2002).
    #
    #  * L. N. Trefethen, Approximation Theory and Approximation Practice, SIAM,
    #    2013, chapter 18.
    #
    #  [TODO]: Update this reference.
    #  * Y. Nakatsukasa and V. Noferini, On the stability of polynomial rootfinding
    #  via linearizations in nonmonomial bases, (In Prep).
    #
    ################################################################################
    """

    # Deal with empty case:
    if len(f.coeffs) == 0:
        out = np.array([])
        return out


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
    if f.length() == 1:
        if f.coeffs[0] == 0 and roots_pref['zero_fun']:
            # Return a root at centre of domain:
            out = np.array([0.0])
        else:
            # Return empty:
            out = np.array([])
        return out

    # Get scaled coefficients for the recursive call:
    c = f.coeffs/f.vscale

    # Call the recursive rootsunit function:
    # TODO:  Does the tolerance need to depend on some notion of hscale?
    default_tol = np.spacing(1)*100
    r = rootsunit_coeffs(c, default_tol)

    # Try to filter out spurious roots:
    if roots_pref['filter'] is not None:
        # r = np.sort(r, 'ascend');
        r = np.sort(r, 'ascend');
        fltr = roots_pref['filter']
        r = fltr(r, f)

    # Prune the roots, if required:
    # if ( roots_pref.prune && ~roots_pref.recurse )
    if roots_pref['prune'] and not roots_pref['recurse']:
        rho = np.sqrt(np.spacing(1))**(-1/f.length())
        rho_roots = np.abs(r + np.sqrt(r**2 - 1))
        rho_roots[rho_roots < 1] = 1/rho_roots[rho_roots < 1]
        out = r[rho_roots <= rho]
    else:
        out = r

    return out


def rootsunit_coeffs(c, htol)
    # Computes the roots of the polynomial given by the coefficients c on the
    # unit interval.

        # Define these as persistent, need to compute only once.
        persistent Tleft Tright

        # Simplify the coefficients:
        tailMmax = np.spacing(1)*np.linalg.norm(c, 1);
        # Find the final coefficient about tailMax:
        # n = find(abs(c) > tailMmax, 1, 'last');
        coeffs_index = np.nonzero(np.abs(c) > tailMmax)[0]
        if len(coeffs_index) > 0:
            n = coeffs_index[-1]
        else:
            n = None

        # Should we alias or truncate here? We truncate here for speed (about
        # 30-50# faster on example with a large amount of subdivision. 
        # Wrap (i.e., alias), don't just truncate:
#         if ( ~isempty(n) && (n > 1) && (n < length(c)) )
#             c = chebtech2.alias(c(end:-1:1), n);
#             c = c(end:-1:1);
#         end
        # Truncate the coefficients (rather than alias):
        # if ( ~isempty(n) && (n > 1) && (n < length(c)) )
        if n is not None and n > 0 and n < len(c)-1:
                # c = c(1:n)
                c = c[:n]

        # Trivial case, n == []:
        # if ( isempty(n) )
        if n is None:
            if roots_pref['zero_fun']:
                # If the function is zero, then place a root in the middle:
                r = np.array([0.0])
            else:
                # Else return empty:
                r = np.array([])
        # Trivial case, n == 1:
        elif n == 1:
            # If the function is zero, then place a root in the middle:
            # if ( c(1) == 0 && roots_pref.zero_fun )
            if c[0] == 0.0 and roots_pref['zero_fun']:
                # If the function is zero, then place a root in the middle:
                r = np.array([0.0])
            else:
                # Else return empty:
                r = np.array([])

        # Trivial case, n == 2:
        elif n == 2:
            # Is the root in [-1,1]?
            r = -c[0]/c[1]
            if not roots_pref['all']:
                if  np.abs(r.imag) > htol or r < -(1 + htol) or r > (1 + htol):
                    r = np.array([])
                else:
                    # r = max(min(real(r), 1), -1);
                    r = np.max([np.min([r.real, 1]), -1])

        # Is n small enough for the roots to be calculated directly?
        elif (not roots_pref['recurse']) or (n <= 50):
            # Adjust the coefficients for the colleague matrix:
            c_old = c.copy()
            # c = -0.5 * c(1:end-1) / c(end);
            c = -0.5 * c[0:-1]/c[-1]
            # c(end-1) = c(end-1) + 0.5;
            c[-2] = c[-2] + 0.5
            
            # Modified colleague matrix:
            # [TODO]: Would the upper-Hessenberg form be better?
            # oh = 0.5 * ones(length(c)-1, 1);
            oh = 0.5 * np.ones(len(c)-1)
            A = np.diag(oh, 1) + np.diag(oh, -1);
            # A(end-1, end) = 1;
            A[-2, -1] = 1.0
            # A(:, 1) = flipud( c );
            A[:, 0] = c[::-1]
            
            # Compute roots by an EP if qz is 'off' and by a GEP if qz is 'on'.
            # QZ has been proved to be stable, while QR is not (see [Nakatsukasa
            # & Noferini, 2014]):
            if roots_pref['qz']:
                # Set up the GEP. (This is more involved because we are scaling
                # for extra stability.)
                B = np.eye(len(A))
                c_old = c_old / np.linalg.norm( c_old, np.inf ); 
                B[0, 0] = c_old[-1]
                # cOld = -0.5 * cOld( 1:end-1 );
                c_old = -0.5 * c_old[:-1]
                # cOld( end-1 ) = cOld( end-1 ) + 0.5 * B(1, 1);
                c_old[-2] = c_old[-2] + 0.5 * B[0, 0]
                # A(:, 1) = flipud( cOld ); 
                A[:, 0] = c_old[::-1]
                r = eig(A, B);
            else:
                # Standard colleague (See [Good, 1961]):
                r = numpy.linalg.eig(A)
           
            # Clean the roots up a bit:
            if not roots_pref['all']:
                # Remove dangling imaginary parts:
                # mask = abs(imag(r)) < htol;
                # r = real( r(mask) );
                r = r[np.abs(r.imag) < htol].real
                # Keep roots inside [-1 1]:
                r = np.sort(r[abs(r) <= 1 + htol])
                # Correct roots over ends:
                if len(r) > 0:
                    r[0] = np.max([r[0], -1])
                    r[-1] = np.min([r[-1], 1])
            elif roots_pref['prune']:
                # rho = sqrt(eps)^(-1/n);
                rho = np.spacing(1)**(-1/(2*n))
                # rho_roots = abs(r + sqrt(r.^2 - 1));
                rho_roots = np.abs(r + np.sqrt(r**2 - 1))
                # rho_roots(rho_roots < 1) = 1./rho_roots(rho_roots < 1);
                rho_roots[rho_roots < 1] = 1.0/rho_roots[rho_roots < 1]
                # r = r(rho_roots <= rho);
                r = r[rho_roots <= rho]

        # If n <= 513 then we can compute the new coefficients with a
        # matrix-vector product.
        elif n <= 513:
            
            # Have we assembled the matrices TLEFT and TRIGHT?
            # [TODO] How to make Tleft and Tright persistent?
            # if ( isempty(Tleft) )

            # Create the coefficients for TLEFT using the FFT directly:
            x = chebpts_ab(513, -1, split_point)
            Tleft = np.ones((513, 513))
            Tleft[:,1] = x
            for k in range(2,513):
                Tleft[:,k] = 2 * x * Tleft[:, k-1] - Tleft[:, k-2]
            # Tleft = [ Tleft(513:-1:2,:) ; Tleft(1:512,:) ];
            Tleft = np.r_[ Tleft[-1:0:-1, :], Tleft[0:-1, :]]
            # Tleft = real(fft(Tleft) / 512);
            Tleft = np.fft.fft(Tleft).real/512
            # Tleft = triu( [ 0.5*Tleft(1,:) ; Tleft(2:512,:) ; 0.5*Tleft(513,:) ] );
            Tleft[0, :] = 0.5 * Tleft[0, :]
            Tleft[-1, :] = 0.5 * Tleft[-1, :]
            Tleft = np.triu(Tleft)

            # Create the coefficients for TRIGHT much in the same way:
            x = chebpts_ab(513, split_point,1)
            Tright = np.ones((513, 513); 
            Tright[:,1] = x;
            # for k = 3:513
            for k in range(2, 513)
                # Tright(:,k) = 2 * x .* Tright(:,k-1) - Tright(:,k-2); 
                Tright[:, k] = 2 * x * Tright[:, k-1] - Tright[:, k-2]

            # Tright = [ Tright(513:-1:2,:) ; Tright(1:512,:) ];
            Tright = np.r_[ Tright[-1:0:-1, :], Tright[0:-1, :] ]
            # Tright = real(fft(Tright) / 512);
            Tright = np.fft.fft(Tright).real / 512
            # Tright = triu( [ 0.5*Tright(1,:) ; Tright(2:512,:) ; 0.5*Tright(513,:) ] );
            Tleft[0, :] = 0.5 * Tleft[0, :]
            Tleft[-1, :] = 0.5 * Tleft[-1, :]
            Tleft = np.triu(Tleft)

            # Compute the new coefficients:
            # cleft = Tleft(1:n,1:n) * c
            cleft = np.dot(Tleft[0:n, 0:n], c)
            # cright = Tright(1:n,1:n) * c;
            cright = np.dot(Tright[0:n, 0:n], c)

            # Recurse:
            r = np.r_[ (split_point - 1)/2 + (split_point + 1)/2*rootsunit_coeffs(cleft, 2*htol),
                  (split_point + 1)/2 + (1 - split_point)/2*rootsunit_coeffs(cright, 2*htol) ]

        # Otherwise, split using more traditional methods (i.e., Clenshaw):
        else:
            
            # Evaluate the polynomial on both intervals:
            v = chebtech.clenshaw.clenshaw(np.r_[ chebpts_ab(n, -1, split_point ),
                chebpts_ab(n, split_point, 1 ) ], c);

            # Get the coefficients on the left:
            # cleft = chebtech2.vals2coeffs.vals2coeffs(v(1:n));            
            cleft = chebtech2.vals2coeffs.vals2coeffs(v[0:n])

            # Get the coefficients on the right:
            # cright = chebtech2.vals2coeffs.vals2coeffs(v(n+1:end));           
            cright = chebtech2.vals2coeffs.vals2coeffs(v[n:])

            # Recurse:
            r = np.r_[ (split_point - 1)/2 + (split_point + 1)/2*rootsunit_coeffs(cleft, 2*htol),
                  (split_point + 1)/2 + (1 - split_point)/2*rootsunit_coeffs(cright, 2*htol) ]

    return r





def chebpts_ab(n, a, b)
    """
    # Y = CHEBPTS_AB(N, A, B) is the N-point Chebyshev grid mapped to [A,B].
    """
    x = chebtech2.chebpts(n);          # [-1,1] grid
    y = b*(x + 1)/2 + a*(1 - x)/2;     # new grid
    return y

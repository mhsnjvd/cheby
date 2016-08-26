import numpy as np
import chebtech2.coeffs2vals

def cumsum(g):
    """
    #CUMSUM   Indefinite integral of a CHEBTECH.
    #   CUMSUM(G) is the indefinite integral of the CHEBTECH G with the constant of
    #   integration chosen so that G(-1) = 0.
    #
    #   CUMSUM(G, 2) will take cumulative sum over the columns of G which is an
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

    f = g.copy()

    # Trivial case of an empty CHEBTECH:
    if f.length() == 0:
        return f


    # Initialise storage:
    # Obtain Chebyshev coefficients {c_r}
    c = f.coeffs                      

    # [n, m] = size(c);
    n = len(c)

    # Pad with zeros
    # c = [ c ; zeros(2, m) ;];
    c = np.r_[c, np.zeros(2)]

    # Initialize vector b = {b_r}
    # b = zeros(n+1, m);
    b = np.zeros(n+1)

    # Compute b_(2) ... b_(n+1):
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

    # Recover coeffs:
    f.coeffs = b
    f.values = chebtech2.coeffs2vals.coeffs2vals(b)

    # Simplify (as suggested in Chebfun ticket #128)
    # f = simplify(f);

    # Ensure f(-1) = 0:
    # lval = get(f, 'lval');
    # f.coeffs(1,:) = f.coeffs(1,:) - lval;

    return f

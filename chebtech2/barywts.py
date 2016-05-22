import numpy as np

def barywts(n):
    """
    %BARYWTS   Barycentric weights for Chebyshev points of 2nd kind.
    %   BARYWTS(N) returns the N barycentric weights for polynomial interpolation on
    %   a Chebyshev grid of the 2nd kind. The weights are normalised so that they
    %   have infinity norm equal to 1 and the final entry is positive.
    %
    % See also BARY, CHEBPTS.   

    % Copyright 2016 by The University of Oxford and The Chebfun Developers.
    % See http://www.chebfun.org/ for Chebfun information.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % See Thm. 5.2 of Trefethen, Approximation Theory and Approximation Practice, 
    % SIAM, 2013 for more information.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    # Special case (no points)
    if n == 0:
        v = []
    elif n == 1:
        # Special case (single point)
        v = 1
    else:
        # Note v(end) is positive.
        v = np.r_[np.ones(n-1), .5] 
        v[-2::-2] = -1.0
        v[0] = .5 * v[0]
    return v

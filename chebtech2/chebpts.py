import numpy as np

import angles, barywts, quadwts

def chebpts(n):
    """
    %CHEBPTS   Chebyshev points in [-1, 1].
    %   CHEBPTS(N) returns N Chebyshev points of the 2nd kind in [-1,1].
    %
    %   [X, W] = CHEBPTS(N) returns also a row vector of the weights for
    %   Clenshaw-Curtis quadrature (computed using Waldvogel's method: [1], [2]).
    %
    %   [X, W, V] = CHEBPTS(N) returns, in addition to X and W, the barycentric
    %   weights V corresponding to the Chebyshev points X. The barycentric weights
    %   are normalised to have infinity norm equal to 1 and a positive first entry.
    %
    %   [X, W, V, T] = CHEBPTS(N) returns also the angles of X.
    %
    % See also BARY, QUADWTS, BARYWTS, TRIGPTS, LEGPTS, JACPTS, LAGPTS, and HERMPTS.

    % Copyright 2016 by The University of Oxford and The Chebfun Developers. 
    % See http://www.chebfun.org/ for Chebfun information.
    """

    # Special case (no points)
    if n == 0:
        x = []
        w = [] 
        v = []
        t = []
    # Special case (single point)
    elif n == 1:
        x = 0 
        w = 2 
        v = 1 
        t = np.pi/2
        
    else:
        # Chebyshev points:
        m = n - 1
        # (Use of sine enforces symmetry.)
        x = np.sin(np.pi*(np.r_[-m:m+1:2]/(2*m)))
        
        # [TODO] how to do a nargout in python?
        # Quadrature weights:            
        w = quadwts.quadwts(n)
        
        # Barycentric weights:
        v = barywts.barywts(n)
        
        # Angles:
        t = angles.angles(n);

    return x, w, v, t

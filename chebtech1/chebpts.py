import numpy as np

import angles, barywts, quadwts

def chebpts(n):
    """
    %CHEBPTS   Chebyshev points of 1st kind in [-1, 1].
    %   CHEBPTS(N) returns N Chebyshev points of the 1st kind in [-1, 1].
    %
    %   [X, W] = CHEBPTS(N) returns also a row vector of the weights for
    %   Clenshaw-Curtis quadrature (computed using [1,2] ).
    %
    %   [X, W, V] = CHEBPTS(N) returns, in addition to X and W, the weights V
    %   for barycentric polynomial interpolation in the Chebyshev points X.
    %
    %   [X, W, V, T] = CHEBPTS(N) returns also the angles of X.
    %
    % See also ANGLES, TRIGPTS, LEGPTS, JACPTS, LAGPTS, and HERMPTS.

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
    # General case
    else:
        # Chebyshev points using sine function to preserve symmetry:
        x = np.sin(np.pi*(np.r_[-n+1:n:2]/(2*n)))
        
        # [TODO] how to do a nargout in python?
        # Quadrature weights:            
        # w = chebtech1.quadwts(n);
        w = quadwts.quadwts(n)
        
        # Barycentric weights:
        # v = chebtech1.barywts(n);
        v = barywts.barywts(n)
        
        # Angles:
        t = angles.angles(n);

    return x, w, v, t

import numpy as np

def angles(n):
    """
    %ANGLES   Return the angles of the Chebyshev points of 1st kind in [-1, 1].
    %   CHEBTECH1.ANGLES(N) returns ACOS(X), where X are the N Chebyshev points of
    %   the 1st kind in [-1, 1].
    %
    % See also POINTS, CHEBPTS, LENGTH.
    %
    % Copyright 2016 by The University of Oxford and The Chebfun Developers.
    % See http://www.chebfun.org/ for Chebfun information.
    """

    out = np.flipud(np.r_[.5:n+.5]) * np.pi/n
    return out

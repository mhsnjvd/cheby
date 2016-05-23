import numpy as np
from chebtech2 import chebpts, barywts
import num2nparray

def bary(*args):
    """
    #BARY   Barycentric interpolation formula.
    #   BARY(X, FVALS, XK, VK) uses the 2nd form barycentric formula with weights VK
    #   to evaluate an interpolant of the data {XK, FVALS(:,k)} at the points X.
    #   Note that XK and VK should be column vectors, and FVALS, XK, and VK should
    #   have the same length.
    #
    #   BARY(X, FVALS) assumes XK are the 2nd-kind Chebyshev points and VK are the
    #   corresponding barycentric weights.
    #
    #
    #   Example:
    #     x = chebpts(181);
    #     f = 1./( 1 + 25*x.^2 );
    #     xx = linspace(-1, 1, 1000);
    #     [xx, yy] = meshgrid(xx, xx);
    #     ff = bary(xx + 1i*yy, f);
    #     h = surf(xx, yy, 0*xx, angle(-ff));
    #     set(h, 'edgealpha', 0)
    #     view(0, 90), shg
    #     colormap(hsv)
    #
    # See also CHEBTECH.CLENSHAW.

    # Copyright 2016 by The University of Oxford and The Chebfun Developers.
    # See http://www.chebfun.org/ for Chebfun information.
    """

    # Parse inputs:
    x = num2nparray.num2nparray(args[0])
    fvals = num2nparray.num2nparray(args[1])

    n = len(fvals)

    if len(args) < 4:
        xk = chebpts.chebpts(n)[0]
        vk = barywts.barywts(n)
    else:
        xk = args[2]
        vk = args[3]


    # Trivial case
    if len(x) == 0:
        fx = x.copy()
        return fx

    # The function is a constant.
    if n == 1:
        fx = fvals * np.ones(len(x))
        return fx

    # The function is NaN.
    if any(np.isnan(fvals)):
        fx = np.nan * np.ones(len(x))
        return fx

    # The main loop:
    # Ignore divide by 0 warning:
    # [TODO] how to restore the warning state?
    np.seterr(divide='ignore', invalid='ignore')
    if len(x) < 4*len(xk):
        # Loop over evaluation points
        # Note: The value "4" here was determined experimentally.

        # Initialise return value:
        fx = np.zeros(len(x))

        # Loop:
        for j in range(0, len(x)):
            xx = vk / (x[j] - xk);
            fx[j] = np.dot(xx, fvals) / xx.sum()
    else:                           
        # Loop over barycentric nodes
        # Initialise:
        num = np.zeros(len(x))
        denom = np.zeros(len(x))

        # Loop:
        for j in range(0, len(xk)):
            tmp = vk[j] / (x - xk[j])
            num = num + tmp * fvals[j]
            denom = denom +  tmp

        fx = num / denom;

    # Try to clean up NaNs:
    for k in np.nonzero(np.isnan(fx))[0]:
        # (Transpose as Matlab loops over columns)
        index = np.nonzero(xk == x[k])[0]
        if len(index) > 0:
            fx[k] = fvals[index[0]]

    return fx

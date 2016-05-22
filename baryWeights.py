import numpy as np
def baryWeights(x):
    """
    BARYWEIGHTS   Barycentric weights.
       W = BARYWEIGHTS(X) returns scaled barycentric weights for the points in the
       column vector X. The weights are scaled such that norm(W, inf) == 1.

    Copyright 2016 by The University of Oxford and The Chebfun Developers. 
    See http://www.chebfun.org/ for Chebfun information.

    [TODO]: Should this live in the trunk?
    """

    # input dimension:
    n = len(x)

    # Capacity:
    if np.isreal(x).all():
        C = 4/(x.max() - x.min())   # Capacity of interval.
    else:
        C = 1 # Scaling by capacity doesn't apply for complex nodes.

    # Compute the weights:
    w = np.ones(n)
    for j in range(0, n):
        v = C*(x[j] - x)
        v[j] = 1
        vv = np.exp(np.log(np.abs(v)).sum());
        w[j] = 1/(np.prod(np.sign(v))*vv)

    # Scaling:
    w = w/np.linalg.norm(w, np.inf)

    return w

import numpy as np

def clenshaw(x, c):
    """
    #CLENSHAW   Clenshaw's algorithm for evaluating a Chebyshev expansion.
    #   If C is a column vector, Y = CLENSHAW(X, C) evaluates the Chebyshev
    #   expansion
    #
    #     Y = P_N(X) = C(1)*T_0(X) + ... + C(N)*T_{N-1}(X) + C(N+1)*T_N(X)
    #
    #   using Clenshaw's algorithm.
    #
    #   X must be a column vector.
    #
    # See also CHEBTECH.FEVAL, CHEBTECH.BARY.

    # Copyright 2016 by The University of Oxford and The Chebfun Developers. 
    # See http://www.chebfun.org/ for Chebfun information.

    ################################################################################
    # Developer note: Clenshaw is not typically called directly, but by FEVAL().
    #
    # Developer note: The algorithm is implemented both for scalar and for vector
    # inputs. Of course, the vector implementation could also be used for the scalar
    # case, but the additional overheads make it a factor of 2-4 slower. Since the
    # code is short, we live with the minor duplication.
    #
    ################################################################################
    """

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

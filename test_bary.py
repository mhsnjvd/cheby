import numpy as np
import bary
from chebtech2 import chebpts, barywts

import matplotlib.pyplot as plt

def test_bary(*args):
    fx = bary.bary(*args)
    print(fx)
    x = args[0]
    fvals = args[1]
    if len(args) > 2:
        xk = args[2]
    else:
        xk = chebpts.chebpts(len(fvals))[0]

    plt.plot(x, fx, 'b')
    plt.plot(xk, fvals, 'ro')
    a = fx.min()
    b = fx.max()
    d = .4*(b-a)
    plt.axis([-1, 1, a-d, b+d])
    plt.show()
    

if __name__ == "__main__":
    x = np.linspace(-1, 1, 10)
    fvals = np.array([-1, 0, 1])
    xk = chebpts.chebpts(len(fvals))[0]
    vk = barywts.barywts(len(fvals)) 
    test_bary(x, fvals, xk, vk)
    test_bary(x, fvals)

    # Evaluate a chebyshev polynomial
    # on a different grid: 
    n = 10
    x = np.linspace(-1, 1, 2001)
    fvals = np.ones(n)
    fvals[::2] = -1.0
    test_bary(x, fvals)

import numpy as np
import bary
from chebtech2 import chebpts, barywts

def test_bary(x, fvals, xk, vk):
    fx = bary.bary(x, fvals, xk, vk)
    print(fx)

if __name__ == "__main__":
    x = np.linspace(-1, 1, 3)
    fvals = np.array([1, 0, 1])
    xk = chebpts.chebpts(len(fvals))
    vk = barywts.barywts(len(fvals)) 
    test_bary(x, fvals, xk, vk)

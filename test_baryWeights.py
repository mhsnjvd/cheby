import numpy as np
import baryWeights
from chebtech2 import chebpts, barywts

def test_baryWeights(x):
    w = baryWeights.baryWeights(x)
    print(x)
    print(w)

if __name__ == "__main__":
    x = np.linspace(-1, 1, 5)
    test_baryWeights(x)
    x = np.linspace(-1, 1, 6)
    test_baryWeights(x)
    x = chebpts.chebpts(4)[0]
    test_baryWeights(x)

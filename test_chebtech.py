import numpy as np
from chebtech.chebtech import Chebtech

f = Chebtech(coeffs=[0, 0, 1])
g = Chebtech(coeffs=[0, 1])
h = f + g
k = g + f
print( h.coeffs)
print( k.coeffs)
print(h[0.1])

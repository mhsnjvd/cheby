import numpy as np
from chebtech.chebtech import Chebtech
from chebtech.cumsum import cumsum
from chebtech.roots import roots


f = Chebtech(coeffs=[0, 0, 1])
g = Chebtech(coeffs=[0, 1])
l = Chebtech(coeffs=[0, 0, 0, 1])
print(roots(l))
h = f + g
k = g + f
print( h.coeffs)
print( k.coeffs)
print( 'f[0] = %s' % f[0] )

#f.plot()
g = cumsum(f)
#g.plot()


r = roots(f)
print(r)

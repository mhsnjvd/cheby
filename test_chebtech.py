import numpy as np
from chebtech.chebtech import Chebtech
#from chebtech.roots import roots

if __name__ == "__main__":
    f = Chebtech(coeffs=[0, 0, 1])
    f.roots()
    print(f.coeffs)
    g = Chebtech(coeffs=[0, 1])
    print('roots of x are:')
    print(g.roots())

    l = Chebtech(coeffs=[0, 0, 0, 1])
    #print(roots(l))
    h = f + g
    k = g + f
    print( h.coeffs)
    print( k.coeffs)
    print( 'f[0] = %s' % f[0] )

    #f.plot()
    g = f.cumsum()
    #g.plot()

    f = Chebtech(coeffs=[1, .75, 0, .25])
    print(f.coeffs)

    #r = roots(f)
    #print(r)

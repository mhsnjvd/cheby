from chebtech.chebtech import Chebtech
import numpy as np
from scipy import fftpack
from scipy import linalg

if __name__ == "__main__":
    n = int(5 + 10*np.random.rand())
    coeffs = np.random.randn(n) + 1.0j * np.random.randn(n)
    v1 = Chebtech.coeffs2vals(coeffs)

    coeffs[1:-1] = 0.5 * coeffs[1:-1]
    coeffs[1::2] = -1.0 * coeffs[1::2]
    v2 = fftpack.idct(coeffs, type=1)
    # c2[1:-1] = 1.0/(len(c2)-1)*c2[1:-1]
    # c2[[0, -1]] = 1.0/(2*(len(c2)-1)) * c2[[0, -1]]
    # c2[1::2] = -1.0*c2[1::2]
    print(v1)
    print(v2)
    print(linalg.norm(v1-v2, np.inf))


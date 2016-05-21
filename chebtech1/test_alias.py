import numpy as np
import alias

def test_alias(c, m):
    print(m)
    print(alias.alias(c, m))



if __name__ == "__main__":
    coeffs = np.r_[1:8]
    print("Aliasing coefficients: %s" % coeffs )
    for m in range(0,len(coeffs)+2):
    # for m in [3, 4, 5, 6]:
        test_alias(coeffs[::], m)


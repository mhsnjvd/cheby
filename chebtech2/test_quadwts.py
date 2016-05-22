import numpy as np
import quadwts

def test_quadwts(n):
    print(n)
    print(quadwts.quadwts(n))

if __name__ == "__main__":
    for n in range(0, 10):
        test_quadwts(n)


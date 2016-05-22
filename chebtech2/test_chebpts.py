import numpy as np
import chebpts

def test_chebpts(n):
    print(n)
    print(chebpts.chebpts(n)[0])

if __name__ == "__main__":
    for n in range(0, 10):
        test_chebpts(n)


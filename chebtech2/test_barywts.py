import numpy as np
import barywts

def test_barywts(n):
    print(n)
    print(barywts.barywts(n))

if __name__ == "__main__":
    for n in range(0, 10):
        test_barywts(n)


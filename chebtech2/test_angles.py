import numpy as np
import angles

def test_angles(n):
    print(n)
    print(angles.angles(n))

if __name__ == "__main__":
    for n in range(0, 10):
        test_angles(n)


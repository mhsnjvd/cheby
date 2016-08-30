import unittest
import numpy as np
from chebtech.chebtech import Chebtech

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


class TestChebtechMethods(unittest.TestCase):

    def test_ctor(self):
        f = Chebtech()
        self.assertEqual(len(f), 0)
        self.assertTrue(f is not None)
        self.assertFalse(fj

    def test_coeffs2vals(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_vals2coeffs(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChebtechMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)

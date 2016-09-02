import unittest
import numpy as np
from scipy import linalg, special
from chebtech.chebtech import Chebtech

class TestChebtechMethods(unittest.TestCase):

    def test_ctor(self):
        f = Chebtech()
        self.assertEqual(len(f), 0)
        self.assertTrue(f is not None)
        self.assertFalse(isinstance(f, list))

        #% Get preferences:
        #if ( nargin < 1 )
        #    pref = chebtech.techPref();
        #end
        #% Set the tolerance:
        #tol = 100 * np.spacing(1)
        #
        ## Initialize with default data:
        #data = chebtech.parseDataInputs(struct());
        #
        #%%
        #% Test on a scalar-valued function:
        #pref.extrapolate = 0;
        #pref.refinementFunction = 'nested';
        #f = @(x) sin(x);
        #g = populate(chebtech2, f, data, pref);
        #x = chebtech2.chebpts(length(g.coeffs));
        #values = g.coeffs2vals(g.coeffs);
        #pass(1) = norm(f(x) - values, inf) < tol;
        #pass(2) = abs(vscale(g) - sin(1)) < eps && g.ishappy && eps < tol;
        #
        #pref.extrapolate = 1;
        #pref.refinementFunction = 'nested';
        #g = populate(chebtech2, f, data, pref);
        #x = chebtech2.chebpts(length(g.coeffs));
        #values = g.coeffs2vals(g.coeffs);
        #pass(3) = norm(f(x) - values, inf) < tol;
        #pass(4) = norm(vscale(g) - sin(1), inf) < tol && logical(eps);
        #
        #pref.extrapolate = 0;
        #pref.refinementFunction = 'resampling';
        #g = populate(chebtech2, f, data, pref);
        #x = chebtech2.chebpts(length(g.coeffs));
        #values = g.coeffs2vals(g.coeffs);
        #pass(5) = norm(f(x) - values, inf) < tol;
        #pass(6) = abs(vscale(g) - sin(1)) < eps && logical(eps);
        #
        #pref.extrapolate = 1;
        #pref.refinementFunction = 'resampling';
        #g = populate(chebtech2, f, data, pref);
        #x = chebtech2.chebpts(length(g.coeffs));
        #values = g.coeffs2vals(g.coeffs);
        #pass(7) = norm(f(x) - values, inf) < tol;
        #pass(8) = norm(vscale(g) - sin(1), inf) < tol && logical(eps);
        #
        #%%
        #% Test on an array-valued function:
        #pref.extrapolate = 0;
        #pref.refinementFunction = 'nested';
        #f = @(x) [sin(x) cos(x) exp(x)];
        #g = populate(chebtech2, f, data, pref);
        #x = chebtech2.chebpts(length(g.coeffs));
        #values = g.coeffs2vals(g.coeffs);
        #pass(9) = norm(f(x) - values, inf) < tol;
        #
        #pref.extrapolate = 1;
        #pref.refinementFunction = 'nested';
        #g = populate(chebtech2, f, data, pref);
        #x = chebtech2.chebpts(length(g.coeffs));
        #values = g.coeffs2vals(g.coeffs);
        #pass(10) = norm(f(x) - values, inf) < tol;
        #
        #pref.extrapolate = 0;
        #pref.refinementFunction = 'resampling';
        #g = populate(chebtech2, f, data, pref);
        #x = chebtech2.chebpts(length(g.coeffs));
        #values = g.coeffs2vals(g.coeffs);
        #pass(11) = norm(f(x) - values, inf) < tol;
        #
        #pref.extrapolate = 1;
        #pref.refinementFunction = 'resampling';
        #g = populate(chebtech2, f, data, pref);
        #x = chebtech2.chebpts(length(g.coeffs));
        #values = g.coeffs2vals(g.coeffs);
        #pass(12) = norm(f(x) - values, inf) < tol;
        #
        #%%
        #% Some other tests:
        #
        #% This should fail with an error:
        #try
        #    f = @(x) x + NaN;
        #    populate(chebtech2, f, data, pref);
        #    pass(13) = false;
        #catch ME
        #    pass(13) = strcmp(ME.message, 'Too many NaNs/Infs to handle.');
        #end
        #
        #% As should this:
        #try
        #    f = @(x) x + Inf;
        #    populate(chebtech2, f, data, pref);
        #    pass(14) = false;
        #catch ME
        #    pass(14) = strcmp(ME.message, 'Too many NaNs/Infs to handle.');
        #end
        #
        #% Test that the extrapolation option avoids endpoint evaluations.
        #pref.extrapolate = 1;
        #try
        #    populate(chebtech2, @(x) [F(x) F(x)], data, pref);
        #    pass(15) = true;
        #catch ME %#ok<NASGU>
        #    pass(15) = false;
        #end
        #pref.extrapolate = 0;
        #
        #    function y = F(x)
        #        if ( any(abs(x) == 1) )
        #            error('Extrapolate should prevent endpoint evaluation.');
        #        end
        #        y = sin(x);
        #    end
        #
        #% Check that things don't crash if pref.minSamples and pref.maxLength are equal.
        #try
        #    pref.minSamples = 8;
        #    pref.maxLength = 8;
        #    populate(chebtech2, @sin, data, pref);
        #    pass(16) = true;
        #catch
        #    pass(16) = false;
        #end
        #
        #%%
        #% Test logical-valued functions:
        #f = chebtech2(@(x) x > -2);
        #g = chebtech2(1);
        #pass(17) = normest(f - g) < eps;
        #
        #f = chebtech2(@(x) x < -2);
        #g = chebtech2(0);
        #pass(18) = normest(f - g) < eps;
        #
        #end
        

    def test_coeffs2vals(self):
        tol = 100 * np.spacing(1)

        # Test that a single value is converted correctly
        c = np.array([np.sqrt(2)])
        v = Chebtech.coeffs2vals(c)
        self.assertEqual(v, c)

        # Some simple data 
        c = np.r_[1:6]
        # Exact coefficients
        vTrue = np.array([ 3, -4+np.sqrt(2), 3, -4-np.sqrt(2), 15])

        # Test real branch
        v = Chebtech.coeffs2vals(c)
        self.assertTrue(linalg.norm(v - vTrue, np.inf) < tol)
        self.assertFalse(np.any(v.imag))

        # Test imaginary branch
        v = Chebtech.coeffs2vals(1j*c);
        self.assertTrue(linalg.norm(v - 1j*vTrue, np.inf) < tol)
        self.assertFalse(np.any(v.real))

        # Test general branch
        v = Chebtech.coeffs2vals((1+1j)*c)
        self.assertTrue(linalg.norm(v - (1+1j)*vTrue, np.inf) < tol)

        # Test for symmetry preservation
        c = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        v = Chebtech.coeffs2vals(c);
        self.assertTrue(linalg.norm(v - np.flipud(v), np.inf) == 0.0)
        c = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        v = Chebtech.coeffs2vals(c);
        self.assertTrue(linalg.norm(v + np.flipud(v), np.inf) == 0.0)

    def test_vals2coeffs(self):
        tol = 100 * np.spacing(1)

        # Test that a single value is converted correctly
        v = np.array([np.sqrt(2)])
        c = Chebtech.vals2coeffs(v)
        self.assertEqual(v, c)

        # Some simple data 
        v = np.r_[1:6]
        # Exact coefficients
        cTrue = np.array([ 3, 1 + 1/np.sqrt(2), 0, 1 - 1/np.sqrt(2), 0 ])

        # Test real branch
        c = Chebtech.vals2coeffs(v)
        self.assertTrue(linalg.norm(c - cTrue, np.inf) < tol)
        self.assertFalse(np.any(c.imag))

        # Test imaginary branch
        c = Chebtech.vals2coeffs(1j*v);
        self.assertTrue(linalg.norm(c - 1j*cTrue, np.inf) < tol)
        self.assertFalse(np.any(c.real))

        # Test general branch
        c = Chebtech.vals2coeffs((1+1j)*v)
        self.assertTrue(linalg.norm(c - (1+1j)*cTrue, np.inf) < tol)

        # Test for symmetry preservation
        v = np.array([1.1, -2.2, 3.3, -2.2, 1.1])
        c = Chebtech.vals2coeffs(v);
        self.assertTrue(linalg.norm(c[1::2], np.inf) == 0.0)
        v = np.array([1.1, -2.2, 0.0, 2.2, -1.1])
        c = Chebtech.vals2coeffs(v);
        self.assertTrue(linalg.norm(c[::2], np.inf) == 0.0)

    def test_chebpts(self):
        tol = 10 * np.spacing(1)
        
        # Test that n = 0 returns empty results:
        x = Chebtech.chebpts(0)
        self.assertEqual(x.size, 0)
        
        # Test n = 1:
        x = Chebtech.chebpts(1)
        self.assertEqual(len(x), 1)
        self.assertEqual(x[0], 0.0)
        
        # Test that n = 2 returns [-1 , 1]:
        x = Chebtech.chebpts(2);
        self.assertEqual(len(x), 2)
        self.assertTrue(np.all(x == np.array([-1.0, 1.0])))
        
        # Test that n = 3 returns [-1, 0, 1]:
        x = Chebtech.chebpts(3);
        self.assertEqual(len(x), 3)
        self.assertTrue(np.all(x == np.array([-1.0, 0.0, 1.0])))
        
        # % Test that n = 129 returns vectors of the correct size:
        n = 129;
        x = Chebtech.chebpts(n);
        self.assertEqual(len(x), n)
        self.assertTrue(linalg.norm(x[0:int((n-1)/2)] + np.flipud(x[int((n+1)/2):]), np.inf) == 0.0 )
        self.assertEqual(x[int((n-1)/2)], 0.0)

    def test_alias(self):

        tol = 100 * np.spacing(1)

        # Testing a vector of coefficients.
        c0 = np.r_[10.0:0.0:-1]

        # Padding:
        c1 = Chebtech.alias(c0, 11);
        self.assertTrue(linalg.norm(np.r_[c0, 0.0] - c1, np.inf) == 0.0)

        # Aliasing:
        c2 = Chebtech.alias(c0, 9);
        self.assertTrue(linalg.norm(np.r_[10.0:3.0:-1, 4.0, 2.0] - c2, np.inf) == 0.0)
        c3 = Chebtech.alias(c0, 3);
        self.assertTrue(linalg.norm(np.array([18.0, 25.0, 12.0]) - c3, np.inf) == 0.0)

        # Compare against result of evaluating on a smaller grid:
        v = Chebtech.clenshaw(Chebtech.chebpts(9), c0) 
        self.assertTrue(linalg.norm(Chebtech.vals2coeffs(v) - c2, np.inf) < tol)

        v = Chebtech.clenshaw(Chebtech.chebpts(3), c0) 
        self.assertTrue(linalg.norm(Chebtech.vals2coeffs(v) - c3, np.inf) < tol)

        # 
        # Test aliasing a large tail.
        c0 = 1/np.r_[1.0:1001.0]**5
        n = 17;
        c1 = Chebtech.alias(c0, n);
        self.assertEqual(len(c1), n)
        # This should give the same result as evaluating via clenshaw
        v0 = Chebtech.coeffs2vals(c0);
        v2 = Chebtech.clenshaw(Chebtech.chebpts(n), c0);
        c2 = Chebtech.vals2coeffs(v2);
        self.assertTrue(linalg.norm(c1 - c2, np.inf) < n*tol)


    def test_roots(self):

        func = lambda x: (x+1)*50;
        f = Chebtech(lambda x: special.j0(func(x)))
        r = func(roots(f));
        exact = np.array([\
            2.40482555769577276862163, 5.52007811028631064959660, \
            8.65372791291101221695437, 11.7915344390142816137431, \
            14.9309177084877859477626, 18.0710639679109225431479, \
            21.2116366298792589590784, 24.3524715307493027370579, \
            27.4934791320402547958773, 30.6346064684319751175496, \
            33.7758202135735686842385, 36.9170983536640439797695, \
            40.0584257646282392947993, 43.1997917131767303575241, \
            46.3411883716618140186858, 49.4826098973978171736028, \
            52.6240518411149960292513, 55.7655107550199793116835, \
            58.9069839260809421328344, 62.0484691902271698828525, \
            65.1899648002068604406360, 68.3314693298567982709923, \
            71.4729816035937328250631, 74.6145006437018378838205, \
            77.7560256303880550377394, 80.8975558711376278637723, \
            84.0390907769381901578795, 87.1806298436411536512617, \
            90.3221726372104800557177, 93.4637187819447741711905, \
            96.6052679509962687781216, 99.7468198586805964702799])

        self.assertTrue(linalg.norm(r-exact, np.inf) < 1e1 * len(f) * np.spacing(1))
         

        k = 500;
        f = Chebtech(fun=lambda x: np.sin(np.pi*k*x))
        r = f.roots()
        self.assertTrue(linalg.norm(r-(1.0*np.r_[-k:k+1])/k, np.inf) < 1e1 * len(f) * np.spacing(1))

        # Test a perturbed polynomial:
        f = Chebtech(fun=lambda x: (x-.1)*(x+.9)*x*(x-.9) + 1e-14*x**5)
        r = f.roots();
        self.assertEqual(len(r), 4)
        self.assertEqual(linalg.norm(f(r), np.inf) < 1e2*len(f)*np.spacing(1))
        
        
        # Test a some simple polynomials:
        # f = testclass.make([-1 ; 1], [], pref);
        # r = f.roots()
        # pass(n, 4) = all( r == 0 );

        # f = testclass.make([1 ; 0 ; 1]);
        # r = f.roots()
        # pass(n, 5) = numel(r) == 2 && (norm(r, inf) < eps);

        # Test some complex roots:
        # f = Chebtech(fun=lambda x: 1 + 25*x**2)
        # r = f.roots(complex=True)
        # self.assertTrue(linalg.norm( r - [1.0j ; -1.0j]/5.0, np.inf) < 10*np.spacing(1))
            
        # f = Chebtech(fun=lambda x: (1 + 25*x**2)*np.exp(x))
        # r = f.roots(complex=True, prune=True)
        # self.assertTrue(linalg.norm( r - [1.0j ; -1.0j]/5.0, np.inf) < 10*len(f)*np.spacing(1))

        # f = testclass.make(@(x) sin(100*pi*x));
        # r1 = f.roots(complex=True, recurse=False);
        # r2 = f.roots(complex=True);

        # self.assertEqual(len(r1), 201)
        # self.assertEqual(len(r2), 213)

        # Adding test for 'qz' flag: 
        # f = Chebtech(fun=lambda x: 1e-10*x**3 + x**2 - 1e-12)
        # r = f.roots(qz=True)
        # self.assertFalse(len(r)==0)
        # self.assertTrue(linalg.norm(f[r], np.inf) < 10*np.spacing(1))

            
        
        # Add a rootfinding test for low degree non-even functions: 
        # f = Chebtech(fun=lambda x: (x-.5)*(x-1/3))
        # r = f.roots(qz=True)
        # self.assertTrue(linalg.norm(f[r], np.inf) < np.spacing(1))


if __name__ == '__main__':
    k = 500;
    f = Chebtech(fun=lambda x: np.sin(np.pi*k*x))
    r = f.roots()
   
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChebtechMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)

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

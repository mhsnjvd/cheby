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

    def test_radd(self):
        # Generate a few random points to use as test values.
        np.random.seed(6178)
        x = -1 + 2.0 * np.random.rand(100)

        # A random number to use as an arbitrary additive constant.
        alpha = -0.194751428910640 + 0.079812875412665j

        # Check addition with scalars.
        
        f_op = lambda x: np.sin(x);
        f = Chebtech(f_op)

        # Test the addition of a CHEBTECH F, specified by F_OP, to a scalar ALPHA using
        # a grid of points X in [-1  1] for testing samples.
        g1 = f + alpha
        g2 = alpha + f
        self.assertTrue(g1==g2)
        g_exact = lambda x: f_op(x) + alpha
        tol = 10 * g1.vscale()*np.spacing(1)
        self.assertTrue(linalg.norm(g1(x) - g_exact(x), np.inf) <= tol)

    def test_add(self):

        # Test the addition of two CHEBTECH objects F and G, specified by F_OP and
        # G_OP, using a grid of points X in [-1  1] for testing samples.
        def test_add_function_to_function(f, f_op, g, g_op, x):
            h1 = f + g
            h2 = g + f
            result_1 = (h1 == h2)
            h_exact = lambda x: f_op(x) + g_op(x)
            tol = 1e4*h1.vscale()*np.spacing(1)
            result_2 = (linalg.norm(h1(x) - h_exact(x), np.inf) <= tol)

            return result_1 and result_2

        # Generate a few random points to use as test values.
        np.random.seed(6178)
        x = -1 + 2.0 * np.random.rand(100)

        # A random number to use as an arbitrary additive constant.
        alpha = -0.194751428283640 + 0.079814485412665j;

        # Check operation in the face of empty arguments.
        
        f = Chebtech()
        g = Chebtech(lambda x: x)
        self.assertEqual(len(f+f), 0)
        self.assertEqual(len(f+g), 0)
        self.assertEqual(len(g+f), 0)
        
        
        # Check addition of two chebtech objects.
        
        f_op = lambda x: np.zeros(len(x));
        f = Chebtech(f_op)
        self.assertTrue(test_add_function_to_function(f, f_op, f, f_op, x))
        
        f_op = lambda x: np.exp(x) - 1.0
        f = Chebtech(f_op)
        
        g_op = lambda x: 1.0/(1.0 + x**2)
        g = Chebtech(g_op)
        self.assertTrue(test_add_function_to_function(f, f_op, g, g_op, x))
        
        g_op = lambda x: np.cos(1e4*x)
        g = Chebtech(g_op)
        self.assertTrue(test_add_function_to_function(f, f_op, g, g_op, x))
        
        
        g_op = lambda t: np.sinh(t*np.exp(2.0*np.pi*1.0j/6.0))
        g = Chebtech(g_op)
        self.assertTrue(test_add_function_to_function(f, f_op, g, g_op, x))
        
        # Check that direct construction and PLUS give comparable results.
        tol = 10*np.spacing(1)
        f = Chebtech(lambda x: x)
        g = Chebtech(lambda x: np.cos(x) - 1.0)
        h1 = f + g
        h2 = Chebtech(lambda x: x + np.cos(x) - 1.0)
        
        self.assertTrue(linalg.norm(h1.coeffs - h2.coeffs, np.inf) < tol)

        #%
        # Check that adding a CHEBTECH to an unhappy CHEBTECH gives an unhappy
        # result.  

        #f = Chebtech(lambda x: np.cos(x+1));    # Happy
        #g = Chebtech(lambda x: np.sqrt(x+1));   # Unhappy
        #h = f + g;  # Add unhappy to happy.
        #self.assertTrue(n, 20) = (~g.ishappy) && (~h.ishappy);
        #h = g + f;  # Add happy to unhappy.
        #self.assertTrue(n, 21) = (~g.ishappy) && (~h.ishappy);



    def test_rsub(self):
        # Check subtraction with scalars.
        np.random.seed(6178)
        # Generate a few random points to use as test values.
        x = -1.0 + 2.0 * np.random.rand(100)

        # A random number to use as an arbitrary additive constant.
        alpha = np.random.randn() + 1.0j*np.random.randn()
        
        f_op = lambda x: np.sin(x)
        f = Chebtech(f_op)

        # Test the subtraction of a CHEBTECH F, specified by F_OP, to and from a scalar
        # ALPHA using a grid of points X in [-1  1] for testing samples.
        g1 = f - alpha
        g2 = alpha - f
        self.assertTrue(g1==g2)
        g_exact = lambda x: f_op(x) - alpha

        #[TODO] can we bring this tolerance down?
        tol = 1.0e2 * g1.vscale() * np.spacing(1)
        self.assertTrue(linalg.norm(g1(x) - g_exact(x), np.inf) <= tol)

    def test_sub(self):
        # Test the subraction of two CHEBTECH objects F and G, specified by F_OP and
        # G_OP, using a grid of points X in [-1  1] for testing samples.
        def test_sub_function_and_function(f, f_op, g, g_op, x):
            h1 = f - g;
            h2 = g - f;
            result_1 = (h1 == (-1*h2))
            h_exact = lambda x: f_op(x) - g_op(x)
            tol = 1e4 * h1.vscale() * np.spacing(1)
            result_2 = (linalg.norm(h1(x) - h_exact(x), np.inf) <= tol)
            return result_1 and result_2

        np.random.seed(6178)
        # Generate a few random points to use as test values.
        x = -1.0 + 2.0 * np.random.rand(100)

        # A random number to use as an arbitrary additive constant.
        alpha = np.random.randn() + 1.0j*np.random.randn()

        # Check operation in the face of empty arguments.
        
        f = Chebtech()
        g = Chebtech(lambda x: x)
        self.assertEqual(len(f-f), 0)
        self.assertEqual(len(f-g), 0)
        self.assertEqual(len(g-f), 0)
            
        
        # Check subtraction of two chebtech objects.
        
        f_op = lambda x: np.zeros(len(x))
        f = Chebtech(f_op)
        self.assertTrue(test_sub_function_and_function(f, f_op, f, f_op, x))
        
        f_op = lambda x: np.exp(x) - 1;
        f = Chebtech(f_op)
        
        g_op = lambda x: 1.0/(1 + x**2)
        g = Chebtech(g_op)
        self.assertTrue(test_sub_function_and_function(f, f_op, g, g_op, x))
        
        g_op = lambda x: np.cos(1e4*x)
        g = Chebtech(g_op)
        self.assertTrue(test_sub_function_and_function(f, f_op, g, g_op, x))
        
        g_op = lambda t: np.sinh(t*np.exp(2.0*np.pi*1.0j/6))
        g = Chebtech(g_op)
        self.assertTrue(test_sub_function_and_function(f, f_op, g, g_op, x))
        
        
        # Check that direct construction and MINUS give comparable results.
        
        tol = 10*np.spacing(1)
        f = Chebtech(lambda x: x)
        g = Chebtech(lambda x: np.cos(x) - 1)
        h1 = f - g
        h2 = Chebtech(lambda x: x - (np.cos(x) - 1))
        h3 = h1 - h2
        self.assertTrue(linalg.norm(h3.coeffs, np.inf) < tol)

        # [TODO]
        # Check that subtracting a CHEBTECH and an unhappy CHEBTECH gives an
        # unhappy result.  

        #f = Chebtech(lambda x: np.cos(x+1))    # Happy
        #g = Chebtech(lambda x: np.sqrt(x+1))   # Unhappy
        #h = f - g;  # Subtract unhappy from happy.
        #self.assertTrue( 20) = (~g.ishappy) && (~h.ishappy);
        #h = g - f;  # Subtract happy from unhappy.
        #self.assertTrue( 21) = (~g.ishappy) && (~h.ishappy);



    def test_rmul(self):
        # Test the multiplication of a CHEBTECH F, specified by F_OP, by a scalar ALPHA
        # Generate a few random points to use as test values.
        np.random.seed(1918)
        x = -1 + 2.0 * np.random.rand(100)

        # Random numbers to use as arbitrary multiplicative constants.
        alpha = -0.114758928283644 + 0.072473485412265j

        # Check multiplication by scalars.
        f_op = lambda x: np.sin(x)
        f = Chebtech(fun=f_op)
        g1 = f * alpha
        g2 = alpha * f
        self.assertTrue(g1==g2)
        g_exact = lambda x: f_op(x) * alpha
        self.assertTrue(linalg.norm(g1(x) - g_exact(x), np.inf) < 10*np.max(g1.vscale()*np.spacing(1)))

    def test_mul(self):


        # Test the multiplication of two CHEBTECH objects F and G, specified by F_OP and
        # G_OP, using a grid of points X in [-1  1] for testing samples.  If CHECKPOS is
        # TRUE, an additional check is performed to ensure that the values of the result
        # are all nonnegative; otherwise, this check is skipped.
        def test_mul_function_by_function(f, f_op, g, g_op, x, check_positive):
            h = f * g
            h_exact = lambda x: f_op(x) * g_op(x)
            tol = 1e4*np.max(h.vscale()*np.spacing(1))
            result_1 = linalg.norm(h(x) - h_exact(x), np.inf) < tol
            result_2 = True
            if check_positive:
                values = h.coeffs2vals(h.coeffs)
                result_2 = np.all(values >= -tol)
            return result_1 and result_2

        # Generate a few random points to use as test values.
        np.random.seed(6178)
        x = -1 + 2.0 * np.random.rand(100)

        # Random numbers to use as arbitrary multiplicative constants.
        alpha = -0.194758928283640 + 0.075474485412665j

        # Check operation in the face of empty arguments.
        
        f = Chebtech()
        g = Chebtech(fun=lambda x: x)
        self.assertEqual(len(f*f), 0)
        self.assertEqual(len(f*g), 0)
        self.assertEqual(len(g*f), 0)
        
        # Check multiplication by constant functions.
        
        f_op = lambda x: np.sin(x);
        f = Chebtech(f_op)
        g_op = lambda x: np.zeros(len(x)) + alpha
        g = Chebtech(g_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, g, g_op, x, False))
        
        # Spot-check multiplication of two chebtech objects for a few test 
        # functions.
        
        f_op = lambda x: np.ones(len(x))
        f = Chebtech(f_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, f, f_op, x, False))
        
        f_op = lambda x: np.exp(x) - 1.0
        f = Chebtech(f_op)
        
        g_op = lambda x: 1.0/(1.0 + x**2)
        g = Chebtech(g_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, g, g_op, x, False))
        
        g_op = lambda x: np.cos(1.0e4*x);
        g = Chebtech(g_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, g, g_op, x, False))
        
        g_op = lambda t: np.sinh(t*np.exp(2.0*np.pi*1.0j/6.0))
        g = Chebtech(g_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, g, g_op, x, False))
        
        
        # Check specially handled cases, including some in which an adjustment for
        # positivity is performed.
        
        f_op = lambda t: np.sinh(t*np.exp(2.0*np.pi*1.0j/6.0));
        f = Chebtech(f_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, f, f_op, x, False))
        
        g_op = lambda t: np.conjugate(np.sinh(t*np.exp(2.0*np.pi*1.0j/6.0)))
        g = f.conj()
        self.assertTrue(test_mul_function_by_function(f, f_op, g, g_op, x, True))
        
        f_op = lambda x: np.exp(x) - 1.0;
        f = Chebtech(f_op)
        self.assertTrue(test_mul_function_by_function(f, f_op, f, f_op, x, True))
        
        # Check that multiplication and direct construction give similar results.
        
        tol = 50*np.spacing(1)
        g_op = lambda x: 1.0/(1.0 + x**2)
        g = Chebtech(g_op)
        h1 = f * g
        h2 = Chebtech(lambda x: f_op(x) * g_op(x))
        #[TODO] implement prolong:
        #h2 = prolong(h2, length(h1));
        #self.assertTrue(linalg.norm(h1.coeffs - h2.coeffs, np.inf) < tol)
        
        #%
        # Check that multiplying a CHEBTECH by an unhappy CHEBTECH gives an unhappy
        # result.  

        #[TODO] implement happiness :)

        #warning off; # Suppress expected warnings about unhappy operations.
        #f = Chebtech(lambda x: cos(x+1));    # Happy
        #g = Chebtech(lambda x: sqrt(x+1));   # Unhappy
        #h = f.*g;  # Multiply unhappy by happy.
        #pass(n, 23) = (~g.ishappy) && (~h.ishappy); ##ok<*BDSCI,*BDLGI>
        #h = g.*f;  # Multiply happy by unhappy.
        #pass(n, 24) = (~g.ishappy) && (~h.ishappy);
        #warning on; # Re-enable warnings.


    def test_roots(self):

        func = lambda x: (x+1)*50;
        f = Chebtech(lambda x: special.j0(func(x)))
        r = func(f.roots())
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

        #[TODO] Try to get the tolerance down to 1.0e2
        self.assertTrue(linalg.norm(r-exact, np.inf) < 1.0e2 * len(f) * np.spacing(1))
         

        k = 500;
        f = Chebtech(fun=lambda x: np.sin(np.pi*k*x))
        r = f.roots()
        self.assertTrue(linalg.norm(r-(1.0*np.r_[-k:k+1])/k, np.inf) < 1e1 * len(f) * np.spacing(1))

        # Test a perturbed polynomial:
        f = Chebtech(fun=lambda x: (x-.1)*(x+.9)*x*(x-.9) + 1e-14*x**5)
        r = f.roots();
        self.assertEqual(len(r), 4)
        self.assertTrue(linalg.norm(f(r), np.inf) < 1e2*len(f)*np.spacing(1))
        
        
        # Test a some simple polynomials:
        f = Chebtech(values=[-1.0, 1.0])
        r = f.roots()
        self.assertTrue(np.all(r == 0))

        # f = testclass.make([1 ; 0 ; 1]);
        f = Chebtech(values=[1.0, 0.0, 1.0])
        r = f.roots()
        self.assertEqual(len(r), 2)
        self.assertTrue(linalg.norm(r, np.inf) < np.spacing(1))

        # Test some complex roots:
        f = Chebtech(fun=lambda x: 1 + 25*x**2)
        r = f.roots(complex_roots=True)
        self.assertEqual(len(r), 2)
        self.assertTrue(linalg.norm( r - np.r_[1.0j, -1.0j]/5.0, np.inf) < 10*np.spacing(1))
            
        #[TODO] This is failing:
        # f = Chebtech(fun=lambda x: (1 + 25*x**2)*np.exp(x))
        # r = f.roots(complex_roots=True, prune=True)
        # self.assertEqual(len(r), 2)
        # self.assertTrue(linalg.norm( r - np.r_[1.0j, -1.0j]/5.0, np.inf) < 10*len(f)*np.spacing(1))

        #[TODO] We get different number of roots
        #f = Chebtech(fun=lambda x: np.sin(100*np.pi*x))
        #r1 = f.roots(complex_roots=True, recurse=False)
        #r2 = f.roots(complex_roots=True)

        #self.assertEqual(len(r1), 201)
        #self.assertEqual(len(r2), 213)

        # Adding test for 'qz' flag: 
        f = Chebtech(fun=lambda x: 1e-10*x**3 + x**2 - 1e-12)
        r = f.roots(qz=True)
        self.assertFalse(len(r)==0)
        self.assertTrue(linalg.norm(f[r], np.inf) < 10*np.spacing(1))

            
        
        # Add a rootfinding test for low degree non-even functions: 
        f = Chebtech(fun=lambda x: (x-.5)*(x-1/3))
        r = f.roots(qz=True)
        self.assertTrue(linalg.norm(f[r], np.inf) < np.spacing(1))


    def test_max(self):
        # Spot-check the results for a given function.
        def spotcheck_max(fun_op, exact_max):
            f = Chebtech(fun=fun_op)
            y = f.max()
            x = f.argmax()
            fx = fun_op(x)

            #[TODO]: Try to get this tolerance down:
            result = (np.all(np.abs(y-exact_max) < 1.0e2*f.vscale()*np.spacing(1))) and (np.all(np.abs(fx-exact_max) < 1.0e2*f.vscale()*np.spacing(1)))

            return result

        # Spot-check the extrema for a few functions.
        self.assertTrue(spotcheck_max(lambda x: ((x-0.2)**3 - (x-0.2) + 1)*1.0/np.cos(x-0.2), 1.884217141925336))
        self.assertTrue(spotcheck_max(lambda x: np.sin(10*x), 1.0))
        # self.assertTrue(spotcheck_max(lambda x: airy, airy(-1)))
        self.assertTrue(spotcheck_max(lambda x:  -1.0/(1.0 + x**2), -0.5))
        self.assertTrue(spotcheck_max(lambda x: (x - 0.25)**3 * np.cosh(x), 0.75**3*np.cosh(1.0)))


        # Test for complex-valued chebtech objects.
        self.assertTrue(spotcheck_max(lambda x: (x - 0.2)*(np.exp(1.0j*(x - 0.2))+1.0j*np.sin(x - 0.2)), -0.434829305372008 + 2.236893806321343j))


    def test_min(self):
        def spotcheck_min(fun_op, exact_min):
            # Spot-check the results for a given function.
            f = Chebtech(fun=fun_op)
            y = f.min()
            x = f.argmin()
            fx = fun_op(x)
            result = ((np.abs(y - exact_min) < 1.0e2*f.vscale()*np.spacing(1)) and (np.abs(fx - exact_min) < 1.0e2*f.vscale()*np.spacing(1)))

            return result
        # Spot-check the extrema for a few functions.

        self.assertTrue(spotcheck_min(lambda x:  -((x-0.2)**3 -(x-0.2) + 1)*1.0/np.cos(x-0.2), -1.884217141925336))
        self.assertTrue(spotcheck_min(lambda x:  -np.sin(10*x), -1.0))
        #self.assertTrue(spotcheck_min(lambda x:  -airy(x), -airy(-1), pref);
        self.assertTrue(spotcheck_min(lambda x:  1.0/(1 + x**2), 0.5))
        self.assertTrue(spotcheck_min(lambda x:  -(x - 0.25)**3.*np.cosh(x), -0.75**3*np.cosh(1.0)))
        
        
        # Test for complex-valued chebtech objects.
        self.assertTrue(spotcheck_min(lambda x: np.exp(1.0j*x)-0.5j*np.sin(x)+x, 0.074968381369117 - 0.319744137826069j))
        


    def test_sum(self):
        #%
        # Spot-check integrals for a couple of functions.
        f = Chebtech(fun=lambda x: np.exp(x) - 1.0)
        self.assertTrue(np.abs(f.sum() - 0.350402387287603) < 10*f.vscale()*np.spacing(1));

        f = Chebtech(fun=lambda x: 1./(1 + x**2))
        self.assertTrue(np.abs(f.sum() - np.pi/2.0) < 10*f.vscale()*np.spacing(1))

        f = Chebtech(fun=lambda x: np.cos(1e4*x))
        exact = -6.112287777765043e-05
        self.assertTrue(np.abs(f.sum() - exact)/np.abs(exact) < 1e6*f.vscale()*np.spacing(1)) 
        
        z = np.exp(2*np.pi*1.0j/6.0)
        f = Chebtech(fun=lambda t: np.sinh(t*z))
        self.assertTrue(np.abs(f.sum()) < 10*f.vscale()*np.spacing(1))

        # Check a few basic properties.
        a = 2.0
        b = -1.0j
        f = Chebtech(fun=lambda x: x * np.sin(x**2) - 1)
        df = f.diff()
        g = Chebtech(lambda x: np.exp(-x**2))
        dg = g.diff()
        fg = f*g
        gdf = g*df
        fdg = f*dg

        tol_f = 10*f.vscale()*np.spacing(1)
        tol_g = 10*f.vscale()*np.spacing(1)
        tol_df = 10*df.vscale()*np.spacing(1)
        tol_dg = 10*dg.vscale()*np.spacing(1)
        tol_fg = 10*fg.vscale()*np.spacing(1)
        tol_fdg = 10*fdg.vscale()*np.spacing(1)
        tol_gdf = 10*gdf.vscale()*np.spacing(1)

        # Linearity.
        self.assertTrue(np.abs((a*f + b*g).sum() - (a*f.sum() + b*g.sum())) < max(tol_f, tol_g))

        # Integration-by-parts.
        self.assertTrue(np.abs(fdg.sum() - (fg(1) - fg(-1) - gdf.sum())) < np.max(np.r_[tol_fdg, tol_gdf, tol_fg]))

        # Fundamental Theorem of Calculus.
        self.assertTrue(np.abs(df.sum() - (f(1) - f(-1))) < np.max(np.r_[tol_df, tol_f]))
        self.assertTrue(np.abs(dg.sum() - (g(1) - g(-1))) < np.max(np.r_[tol_dg, tol_g]))


    def test_cumsum(self):
        # Generate a few random points to use as test values.

        np.random.seed(6178)
        x = 2 * np.random.rand(100) - 1;

        # Spot-check antiderivatives for a couple of functions.  We verify that the
        # chebtech antiderivatives match the true ones up to a constant by checking 
        # that the standard deviation of the difference between the two on a large 
        # random grid is small. We also check that feval(cumsum(f), -1) == 0 each 
        # time.
      
        f = Chebtech(fun=lambda x: np.exp(x) - 1.0)
        F = f.cumsum()
        F_ex = lambda x: np.exp(x) - x
        err = np.std(F[x] - F_ex(x))
        tol = 20*F.vscale()*np.spacing(1)
        self.assertTrue(err < tol) 
        self.assertTrue(np.abs(F[-1]) < tol)

        f = Chebtech(fun=lambda x: 1.0/(1.0+x**2))
        F = f.cumsum()
        F_ex = lambda x: np.arctan(x)
        err = np.std(F[x] - F_ex(x))
        tol = 10*F.vscale()*np.spacing(1)
        self.assertTrue(err < tol) 
        self.assertTrue(np.abs(F[-1]) < tol)
        
        f = Chebtech(fun=lambda x: np.cos(1.0e4*x))
        F = f.cumsum()
        F_ex = lambda x: np.sin(1.0e4*x)/1.0e4;
        err = F[x] - F_ex(x);
        tol = 10.0e4*F.vscale()*np.spacing(1)
        self.assertTrue((np.std(err) < tol) and (np.abs(F[-1]) < tol))
        
        z = np.exp(2*np.pi*1.0j/6);
        f = Chebtech(fun=lambda t: np.sinh(t*z))
        F = f.cumsum()
        F_ex = lambda t: np.cosh(t*z)/z
        err = F[x] - F_ex(x);
        tol = 10*F.vscale()*np.spacing(1)
        self.assertTrue((np.std(err) < tol) and (np.abs(F[-1]) < tol))
        
        # Check that applying cumsum() and direct construction of the antiderivative
        # give the same results (up to a constant).
        
        f = Chebtech(fun=lambda x: np.sin(4.0*x)**2)
        F = Chebtech(fun=lambda x: 0.5*x - 0.0625*np.sin(8*x))
        G = f.cumsum()
        err = G - F
        tol = 10*G.vscale()*np.spacing(1)
        values = Chebtech.coeffs2vals(err.coeffs)
        self.assertTrue((np.std(values) < tol) and (np.abs(G[-1]) < tol))
        
        # Check that diff(cumsum(f)) == f and that cumsum(diff(f)) == f up to a 
        # constant.
        
        f = Chebtech(lambda x: x*(x - 1.0)*np.sin(x) + 1.0)
        g = f.cumsum().diff()
        err = f(x) - g(x)
        tol = 10*g.vscale()*np.spacing(1)
        self.assertTrue(linalg.norm(err, np.inf) < 100 * tol)

        h = f.diff().cumsum()
        err = f(x) - h(x)
        tol = 10*h.vscale()*np.spacing(1)
        self.assertTrue((np.std(err) < tol)  and (np.abs(h[-1]) < tol))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChebtechMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)

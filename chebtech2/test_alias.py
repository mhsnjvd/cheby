import numpy as np
import alias
import isequal_numerically
import vals2coeffs
import coeffs2vals


def test_alias(coeffs, m):
    aliased_coeffs = alias.alias(coeffs, m)
    # print('coeffs = %s' % coeffs) 
    # print('%s aliased_coeffs = %s' % m, aliased_coeffs)
    return aliased_coeffs

if __name__ == "__main__":

    pass_list = []

    # Set a tolerance (pref.chebfuneps doesn't matter)
    tol = 100*np.spacing(1)

    # Testing a vector of coefficients.
    c0 = np.r_[10:0:-1]

    # Padding:
    c1 = test_alias(c0, 11)
    test_value =  ( c1[-1] == 0 and (c1[:-1] == c0).all() )
    if test_value:
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)

    # Aliasing:
    c2 = test_alias(c0, 9)
    test_value = isequal_numerically.isequal_numerically(
            c2, np.r_[np.r_[10:3:-1], 4, 2])
    if test_value:
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)

    c3 = test_alias(c0, 3);
    test_value = isequal_numerically.isequal_numerically(
            c3, np.array([18, 25, 12]))
    if test_value:
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)



    ##
    # Test aliasing a large tail.
    ##c0 = 1/(np.r_[1:1001]**5)
    ##n = 17;
    ##c1 = test_alias(c0, n);
    # This should give the same result as evaluating via bary.
    ##v0 = coeffs2vals.coeffs2vals(c0);
    ##v2 = bary.bary(chebtech2.chebpts(n), v0);
    ##c2 = vals2coeffs.vals2coeffs(v2);
    # Check in the infinity norm:
    ##test_value = isequal_numerically.isequal_numerically(c1, c2, n*tol)

    if pass_list.count(True) == len(pass_list):
        print('All tests in %s passed' % test_alias.__name__ )
    else:
        print('%s failed' % test_alias.__name__ )

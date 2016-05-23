import numpy as np
import vals2coeffs
import isequal_numerically

def test_vals2coeffs(values):
    coeffs = vals2coeffs.vals2coeffs(values)
    # print('values = %s' % values) 
    # print('coeffs = %s' % coeffs)
    return coeffs

if __name__ == "__main__":

    # Initialize list for test results
    pass_list = []

    values = np.array([])
    coeffs = test_vals2coeffs(values)
    test_array = (coeffs == values)
    if test_array.all():
        print('pass')
        pass_list.append(True)
    else:
        print('pass')
        pass_list.append(False)

    values = np.array([1])
    coeffs = test_vals2coeffs(values)
    test_array = (coeffs == values)
    if test_array.all():
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)

    values = np.array([1., 1.])
    coeffs = test_vals2coeffs(values)
    test_value = isequal_numerically.isequal_numerically(
            coeffs, np.array([1., 0.]))
    if test_value:
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)

    values = np.array([1., -1., 1.])
    coeffs = test_vals2coeffs(values)
    test_value = isequal_numerically.isequal_numerically(
            coeffs, np.array([0, 0, 1]))
    if test_value:
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)

    values = np.array([-1., 1., -1., 1.])
    coeffs = test_vals2coeffs(values)
    test_value = isequal_numerically.isequal_numerically(
            coeffs, np.array([0, 0, 0, 1]))
    if test_value:
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)

    if pass_list.count(True) == len(pass_list):
        print('All tests in %s passed' % test_vals2coeffs.__name__ )
    else:
        print('%s failed' % test_vals2coeffs.__name__ )

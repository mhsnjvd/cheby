import numpy as np
import coeffs2vals

def test_coeffs2vals(coeffs):
    values = coeffs2vals.coeffs2vals(coeffs)
    # print('coeffs = %s' % coeffs)
    # print('values = %s' % values) 
    return  values

if __name__ == "__main__":

    # Initialize list for test results
    pass_list = []

    coeffs = np.array([])
    values = test_coeffs2vals(coeffs)
    test_array = (values == coeffs)
    if test_array.all():
        print('pass')
        pass_list.append(True)
    else:
        print('pass')
        pass_list.append(False)

    coeffs = np.array([1])
    values = test_coeffs2vals(coeffs)
    test_array = (values == coeffs)
    if test_array.all():
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)

    coeffs = np.array([1., 0.])
    values = test_coeffs2vals(coeffs)
    test_array = (values == np.array([1., 1.]))
    if test_array.all():
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)


    coeffs = np.array([0, 0, 1])
    values = test_coeffs2vals(coeffs)
    test_array = (values == np.array([1., -1., 1.]))
    if test_array.all():
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)

    coeffs = np.array([0, 0, 0, 1])
    values = test_coeffs2vals(coeffs)
    test_array = (values == np.array([-1., 1., -1., 1.]))
    if test_array.all():
        print('pass')
        pass_list.append(True)
    else:
        print('fail')
        pass_list.append(False)

    if pass_list.count(True) == len(pass_list):
        print('All tests in %s passed' % test_coeffs2vals.__name__ )
    else:
        print('%s failed' % test_coeffs2vals.__name__ )

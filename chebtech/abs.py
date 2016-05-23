import numpy as np

def abs(*args):
    """
    #ABS   Absolute value of a CHEBTECH object.
    #   ABS(F) returns the absolute value of F, where F is a CHEBTECH object with no
    #   roots in [-1 1]. If ~isempty(roots(F)), then ABS(F) will return garbage
    #   with no warning. F may be complex.

    #  Copyright 2016 by The University of Oxford and The Chebfun Developers.
    #  See http://www.chebfun.org/ for Chebfun information.
    """
    
    f = args[0]
    if len(args) > 1:
        other_args = args[1:]
    else:
        other_args = []

    if isreal(f) or isreal(1j*f):
        # Convert to values and then compute ABS(). 
        values = f.coeffs2vals(f.coeffs)
        values = np.abs(values)
        f.coeffs = f.vals2coeffs(values)
    else:
        f = compose(f, @abs, [], [], other_args);

    return f

import numpy as np

def coeffs2vals(coeffs):
    """
    #COEFFS2VALS   Convert Chebyshev coefficients to values at Chebyshev points
    #of the 2nd kind.
    #   V = COEFFS2VALS(C) returns the values of the polynomial V(i,1) = P(x_i) =
    #   C(1,1)*T_{0}(x_i) + ... + C(N,1)*T_{N-1}(x_i), where the x_i are
    #   2nd-kind Chebyshev nodes.
    #
    #  Input: coeffs is numpy.ndarray 
    #  Output: values is a numpy.ndarray of the same length as coeffs
    #   
    #
    # See also VALS2COEFFS, CHEBPTS.

    # Copyright 2016 by The University of Oxford and The Chebfun Developers. 
    # See http://www.chebfun.org/ for Chebfun information.

    ################################################################################
    # [Developer Note]: This is equivalent to Discrete Cosine Transform of Type I.
    #
    # [Mathematical reference]: Sections 4.7 and 6.3 Mason & Handscomb, "Chebyshev
    # Polynomials". Chapman & Hall/CRC (2003).
    ################################################################################
    """

    # *Note about symmetries* The code below takes steps to 
    # ensure that the following symmetries are enforced:
    # even Chebyshev COEFFS exactly zero ==> VALUES are exactly odd
    # odd Chebychev COEFFS exactly zero ==> VALUES are exactly even
    # These corrections are required because the MATLAB FFT does not
    # guarantee that these symmetries are enforced.

    # Get the length of the input:
    n = len(coeffs)

    # Trivial case (constant or empty):
    if n <= 1:
        values = coeffs.copy()
        return values

    # check for symmetry
    # isEven = max(abs(coeffs(2:2:end,:)),[],1) == 0;
    # isOdd = max(abs(coeffs(1:2:end,:)),[],1) == 0;

    # Scale the interior coefficients by 1/2:
    coeffs[1:n-1] = coeffs[1:n-1]/2

    # Mirror the coefficients (to fake a DCT using an FFT):
    tmp = np.r_[coeffs , coeffs[n-2:0:-1]]

    if np.isreal(coeffs).all():
        # Real-valued case:
        values = np.fft.fft(tmp).real
    elif np.isreal(1j*coeffs).all():
        # Imaginary-valued case:
        values = 1j*(np.fft.fft(tmp.imag).real)
    else:
        # General case:
        values = np.fft.fft(tmp)

    # Flip and truncate:
    values = values[n-1::-1]

    # [TODO]enforce symmetry
    # values(:,isEven) = (values(:,isEven)+flipud(values(:,isEven)))/2;
    # values(:,isOdd) = (values(:,isOdd)-flipud(values(:,isOdd)))/2;
    return values

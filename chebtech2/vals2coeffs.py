import numpy as np

def vals2coeffs(values):
    """
    #VALS2COEFFS   Convert values at Chebyshev points to Chebyshev coefficients.
    #   C = VALS2COEFFS(V) returns the (N+1)x1 vector of coefficients such that F(x)
    #   = C(1)*T_0(x) + C(2)*T_1(x) + C(N+1)*T_N(x) (where T_k(x) denotes the
    #   k-th 1st-kind Chebyshev polynomial) interpolates the data [V(1) ; ... ;
    #   V(N+1)] at Chebyshev points of the 2nd kind.
    #
    #   Input: values must be of type numpy.ndarray
    #   Output: a numpy.ndarray of the same size as values
    #
    # See also COEFFS2VALS, CHEBPTS.

    # Copyright 2016 by The University of Oxford and The Chebfun Developers. 
    # See http://www.chebfun.org/ for Chebfun information.

    ################################################################################
    # [Developer Note]: This is equivalent to the Inverse Discrete Cosine Transform
    # of Type I.
    #
    # [Mathematical reference]: Section 4.7 Mason & Handscomb, "Chebyshev
    # Polynomials". Chapman & Hall/CRC (2003).
    ################################################################################
    """

    # *Note about symmetries* The code below takes steps to 
    # ensure that the following symmetries are enforced:
    # VALUES exactly even ==> odd Chebyshev COEFFS are exactly zero
    # VALUES exactly odd ==> even Chebyshev COEFFS are exactly zero
    # These corrections are required because the MATLAB FFT does not
    # guarantee that these symmetries are enforced.

    # Get the length of the input:
    n = len(values)

    # Trivial case (constant):
    if n <= 1:
        coeffs = values.copy()
        return coeffs

    # check for symmetry
    # isEven = max(abs(values-flipud(values)),[],1) == 0;
    # isOdd = max(abs(values+flipud(values)),[],1) == 0;

    # Mirror the values (to fake a DCT using an FFT):
    tmp = np.r_[values[n-1:0:-1], values[0:n-1]]

    if np.isreal(values).all():
        # Real-valued case:
        coeffs = np.fft.ifft(tmp)
        coeffs = coeffs.real
    elif np.isreal(1j*values).all():
        # Imaginary-valued case:
        coeffs = np.fft.ifft(tmp.imag)
        coeffs = 1j*coeffs.real
    else:
        # General case:
        coeffs = np.fft.ifft(tmp)

    # Truncate:
    coeffs = coeffs[0:n]

    # Scale the interior coefficients:
    coeffs[1:n-1] = 2*coeffs[1:n-1]

    # adjust coefficients for symmetry
    # [TODO]
    # coeffs(2:2:end,isEven) = 0;
    # coeffs(1:2:end,isOdd) = 0;

    return coeffs

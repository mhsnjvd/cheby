import numpy as np

def vals2coeffs(values):
    """
    %VALS2COEFFS   Convert values at Chebyshev points to Chebyshev coefficients.
    %   C = VALS2COEFFS(V) returns the (N+1)x1 vector of coefficients such that
    %   F(x) = C(1)*T_N(x) + ... + C(N)*T_1(x) + C(N+1)*T_0(x) (where T_k(x)
    %   denotes the k-th 1st-kind Chebyshev polynomial) interpolates the data
    %   [V(1) ; ... ; V(N+1)] at Chebyshev points of the 1st kind. 
    %
    %   If the input V is an (N+1)xM matrix, then C = VALS2COEFFS(V) returns the
    %   (N+1)xM matrix of coefficients C such that F_j(x) = C(1,j)*T_N(x) + ... 
    %   + C(N,j)*T_1(x) + C(N+1)*T_0(x) interpolates [V(1,j) ; ... ; V(N+1,j)]
    %   for j = 1:M.
    %
    % See also COEFFS2VALS, CHEBPTS.

    % Copyright 2016 by The University of Oxford and The Chebfun Developers. 
    % See http://www.chebfun.org/ for Chebfun information.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % [Developer Note]: This is equivalent to Discrete Cosine Transform of Type II.
    %
    % [Mathematical reference]: Section 4.7 Mason & Handscomb, "Chebyshev
    % Polynomials". Chapman & Hall/CRC (2003).
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % *Note about symmetries* The code below takes steps to 
    % ensure that the following symmetries are enforced:
    % VALUES exactly even ==> odd Chebyshev COEFFS are exactly zero
    % VALUES exactly odd ==> even Chebyshev COEFFS are exactly zero
    % These corrections are required because the MATLAB FFT does not
    % guarantee that these symmetries are enforced.
    """

    # Convert list to numpy array:
    if isinstance(values, list):
        values = np.asarray(values)
     

    # Get the length of the input:
    # n = size(values, 1);
    n = values.shape[0]

    # % Trivial case (constant):
    if n <= 1:
        coeffs = values
        return coeffs

    # [TODO] check for symmetry
    # isEven = max(abs(values-flipud(values)),[],1) == 0;
    # isOdd = max(abs(values+flipud(values)),[],1) == 0;
    # isEven = 1 * ( np.abs(values-np.flipud(values)).max(0) == 0 )
    # isOdd =  1 * ( np.abs(values+np.flipud(values)).max(0) == 0 )

    # [TODO] Computing the weight vector often accounts for at least half the cost of this
    # transformation. Given that (a) the weight vector depends only on the length of
    # the coefficients and not the coefficients themselves and (b) that we often
    # perform repeated transforms of the same length, we store w persistently.
    # persistent w
    # if ( size(w, 1) ~= n )
    # if ( w.shape[0] != n ):

    # Pre-compute the weight vector:
    # w = 2*exp(1i*(0:n-1)*pi/(2*n)).';
    w = 2*np.exp(1j*np.r_[0.0:n]*np.pi/(2*n))

    # Mirror the values for FFT:
    # tmp = [values(n:-1:1, :) ; values];
    tmp = np.r_[np.flipud(values), values]
    coeffs = np.fft.ifft(tmp);

    # Truncate, flip the order, and multiply the weight vector:
    # coeffs = bsxfun(@times, w, coeffs(1:n, :));
    coeffs = w * coeffs[0:n]

    # Scale the coefficient for the constant term:
    # coeffs(1,:) = coeffs(1,:)/2;
    coeffs[0] = coeffs[0]/2

    # Post-process:
    if np.isreal(values).all():
        # Real-valued case:
        coeffs = coeffs.real
    elif np.isreal(1j*values).all():
        # Imaginary-valued case:
        coeffs = 1j*coeffs.imag

    
    # adjust coefficients for symmetry
    # coeffs[1::2,isEven] = 0
    # coeffs[::2,isOdd] = 0

    return coeffs

import numpy as np

def quadwts(n):
    """
    %QUADWTS   Quadrature weights for Chebyshev points of 1st kind.
    %   QUADWTS(N) returns the N weights for quadrature on 1st-kind Chebyshev grid.
    %
    % See also CHEBPTS, BARYWTS.

    % Copyright 2016 by The University of Oxford and The Chebfun Developers.
    % See http://www.chebfun.org/ for Chebfun information.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % DEVELOPER NOTE:
    % We use a variant of Waldvogel's algorithm [1], due to Nick Hale. (See below)
    % We note this is similar to Greg Von Winkel's approach, which can be found on
    % the MathWorks File Exchange.
    %
    % Let $f(x) = \sum_{k=0}^nc_kT_k(x)$, then\vspace*{-3pt} }
    %   I(f) = m.'*c
    % where
    %   m = \int_{-1}^1T_k(x)dx = { 2/(1-k^2) : k even
    %                             { 0         : k odd
    %     = m'*inv(TT)*f(x) where TT_{j,k} = T_k(x_j)
    %     = (inv(TT)'*m)'*f(x)
    % Therefore
    %   I(f) = w.'f(x) => w = inv(TT).'*m;
    % Here inv(TT).' is the discrete cosine transform of type III.
    %
    % Furthermore, since odd entries in m are zero, can compute via FFT without
    % doubling up from N to 2N.
    %
    % References:
    %   [1] Joerg Waldvogel, "Fast construction of the Fejer and Clenshaw-Curtis
    %       quadrature rules", BIT Numerical Mathematics 46 (2006), pp 195-202.
    %   [2] Greg von Winckel, "Fast Clenshaw-Curtis Quadrature", 
    %       http://www.mathworks.com/matlabcentral/fileexchange/6911, (2005)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    # Special case (no points!)
    if n == 0:
        w = []
    elif n == 1:                  
        # Special case (single point)
        w = 2;
    # General case
    else:
        # m = 2./[1, 1-(2:2:(n-1)).^2];  % Moments - Exact integrals of T_k (even)
        # Moments - Exact integrals of T_k (even)
        m = 2/np.r_[1, 1-np.r_[2:n:2]**2]
        
        # Mirror the vector for the use of ifft: 
        if n % 2:
            # n is odd
            # c = [m, -m((n+1)/2:-1:2)];
            c = np.r_[m, -m[(n-1)//2:0:-1]]
        else:
            # n is even 
            # c = [m, 0, -m(n/2:-1:2)];  
            c = np.r_[m, 0, -m[n//2-1:0:-1]]

        # weight (rotation) vector
        v = np.exp(1j*np.r_[0:n]*np.pi/n) 
        # Apply the weight vector
        c = c*v                      
        # Call ifft
        w = np.fft.ifft(c).real             
        
    return w

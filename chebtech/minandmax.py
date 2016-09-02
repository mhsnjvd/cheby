def minandmax(self):
    """Global minimum and maximum on [-1,1].
    #   VALS = MINANDMAX(F) returns a 2-vector VALS = [MIN(F); MAX(F)] with the
    #   global minimum and maximum of the CHEBTECH F on [-1,1].  If F is a
    #   array-valued CHEBTECH, VALS is a 2-by-N matrix, where N is the number of
    #   columns of F.  VALS(1, K) is the global minimum of the Kth column of F on
    #   [-1, 1], and VALS(2, K) is the global maximum of the same.
    #
    #   [VALS, POS] = MINANDMAX(F) returns also the 2-vector POS where the minimum
    #   and maximum of F occur.
    #
    #   If F is complex-valued the absolute values are taken to determine extrema
    #   but the resulting values correspond to those of the original function. That
    #   is, VALS = FEVAL(F, POS) where [~, POS] = MINANDMAX(ABS(F)). (In fact,
    #   MINANDMAX actually computes [~, POS] = MINANDMAX(ABS(F).^2), to avoid
    #   introducing singularities to the function).
    #
    # See also MIN, MAX.

    # Copyright 2016 by The University of Oxford and The Chebfun Developers.
    # See http://www.chebfun.org/ for Chebfun information.
    """

    if not self.isreal():
        # We compute sqrt(max(|f|^2))to avoid intruducing a singularity.
        realf = self.real()
        imagf = self.imag()
        h = realf*realf + imagf.*imagf;
        h = h.simplify()
        [ignored, pos] = h.minandmax(); ##ok<ASGLU>
        vals = f[pos]
        # FEVAL() will not return a matrix argument of the correct dimensions if f
        # is array-valued. This line corrects this:
        vals = vals(:, 1:(size(pos, 2)+1):end);
        vals = vals(:, 1:(size(pos, 2)+1):end);
        return vals

    # Compute derivative:
    fp = self.diff();

    # Make the Chebyshev grid (used in minandmaxColumn).
    xpts = f.points();


    # Initialise output
    pos = Chebtech.zero_array(2)
    vals = Chebtech.zero_array(2)
    
    if f.length() == 1:
        vals = f[pos]
        return vals, pos
    
    # Compute critical points:
    r = fp.roots()
    r = np.unique(np.r_[-1.0, r, 1.0])
    v = f[r]

    # min
    vals(0) = np.min(v)
    pos(0) = r(np.argmin(v))

    # Take the minimum of the computed minimum and the function values:
    values = f.coeffs2vals(f.coeffs);
    temp = np.r_[vals(0), values]
    vmin = np.min(temp)
    vindex = np.argmin(temp)
    if ( vmin < vals(0) )
        vals(0) = vmin;
        pos(0) = xpts(vindex - 1)

    # max
    vals(1) = np.max(v);
    pos(1) = r(np.argmax(v));

    # Take the maximum of the computed maximum and the function values:
    temp = np.r_[vals(1), values]
    vmax = np.max(temp)
    vindex = np.argmax(temp)
    if ( vmax > vals(1) )
        vals(1) = vmax
        pos(1) = xpts(vindex - 1)

    return vals, pos

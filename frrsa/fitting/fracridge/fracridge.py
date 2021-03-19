"""
"""
# =============================================================================
# Note from P.Kaniuth: the whole package "fracridge", of which this module is a
# part, is the sole work of A. Rokem K. Kay, see their paper "Fractional ridge 
# regression: a fast, interpretable reparameterization of ridge regression" 
# (2020) GigaScience, Volume 9, Issue 12, December 2020, 
# https://doi.org/10.1093/gigascience/giaa133.
#     
# The only changes I implemented are the following:
#     - the function "fracridge" now contains the argument "X_test" and outputs
#         predictions directly.
#     - the function "fracridge" now contains the argument "betas_wanted" to
#         specify whether coefs shall be output at all.
#     - other functions defined in the original module were deleted, since they
#         are not needed in the framework of my project FR-RSA.
# Last updated: 1st of February 2021
# =============================================================================

import numpy as np
from numpy import interp
import warnings

# Module-wide constants
BIG_BIAS = 10e3
SMALL_BIAS = 10e-3
BIAS_STEP = 0.2


__all__ = ["fracridge"]


def _do_svd(X, y, jit=True):
    """
    Helper function to produce SVD outputs
    """
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    # Per default, we'll try to use the jit-compiled SVD, which should be
    # more performant:
    use_scipy = False
    if jit:
        try:
            from ._linalg import svd
        except ImportError:
            warnings.warn("The `jit` key-word argument is set to `True` ",
                          "but numba could not be imported, or just-in time ",
                          "compilation failed. Falling back to ",
                          "`scipy.linalg.svd`")
            use_scipy = True

    # If that doesn't work, or you asked not to, we'll use scipy SVD:
    if not jit or use_scipy:
        from functools import partial
        from scipy.linalg import svd  # noqa
        svd = partial(svd, full_matrices=False)

    if X.shape[0] > X.shape[1]:
        uu, ss, v_t = svd(X.T @ X)
        selt = np.sqrt(ss)
        if y.shape[-1] >= X.shape[0]:
            ynew = (1/selt) @ v_t @ X.T @ y
        else:
            ynew = np.diag(1./selt) @ v_t @ (X.T @ y)

    else:
        # This rotates the targets by the unitary matrix uu.T:
        uu, selt, v_t = svd(X)
        ynew = uu.T @ y

    ols_coef = (ynew.T / selt).T

    return selt, v_t, ols_coef


def fracridge(X, X_test, y, fracs=None, tol=1e-10, jit=True, betas_wanted=False):
    """
    Approximates alpha parameters to match desired fractions of OLS length.
    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix for regression, with n number of
        observations and p number of model parameters.
    y : ndarray, shape (n, b)
        Data, with n number of observations and b number of targets.
    fracs : float or 1d array, optional
        The desired fractions of the parameter vector length, relative to
        OLS solution. If 1d array, the shape is (f,). This input is required
        to be sorted. Otherwise, raises ValueError.
        Default: np.arange(.1, 1.1, .1).
    jit : bool, optional
        Whether to speed up computations by using a just-in-time compiled
        version of core computations. This may not work well with very large
        datasets. Default: True
    Returns
    -------
    coef : ndarray, shape (p, f, b)
        The full estimated parameters across units of measurement for every
        desired fraction.
    alphas : ndarray, shape (f, b)
        The alpha coefficients associated with each solution
    Examples
    --------
    Generate random data:
    >>> np.random.seed(0)
    >>> y = np.random.randn(100)
    >>> X = np.random.randn(100, 10)
    Calculate coefficients with naive OLS:
    >>> coef = np.linalg.inv(X.T @ X) @ X.T @ y
    >>> print(np.linalg.norm(coef))  # doctest: +NUMBER
    0.35
    Call fracridge function:
    >>> coef2, alpha = fracridge(X, y, 0.3)
    >>> print(np.linalg.norm(coef2))  # doctest: +NUMBER
    0.10
    >>> print(np.linalg.norm(coef2) / np.linalg.norm(coef))  # doctest: +NUMBER
    0.3
    Calculate coefficients with naive RR:
    >>> alphaI = alpha * np.eye(X.shape[1])
    >>> coef3 = np.linalg.inv(X.T @ X + alphaI) @ X.T @ y
    >>> print(np.linalg.norm(coef2 - coef3))  # doctest: +NUMBER
    0.0
    """
    if fracs is None:
        fracs = np.arange(.1, 1.1, .1)

    if hasattr(fracs, "__len__"):
        if np.any(np.diff(fracs) < 0):
            raise ValueError("The `frac` inputs to the `fracridge` function ",
                             f"must be sorted. You provided: {fracs}")

    else:
        fracs = [fracs]
    fracs = np.array(fracs)

    nn, pp = X.shape
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    bb = y.shape[-1]
    ff = fracs.shape[0]

    # Calculate the rotation of the data
    selt, v_t, ols_coef = _do_svd(X, y, jit=jit)

    # Set solutions for small eigenvalues to 0 for all targets:
    isbad = selt < tol
    if np.any(isbad):
        warnings.warn("Some eigenvalues are being treated as 0")

    ols_coef[isbad, ...] = 0

    # Limits on the grid of candidate alphas used for interpolation:
    val1 = BIG_BIAS * selt[0] ** 2
    val2 = SMALL_BIAS * selt[-1] ** 2

    # Generates the grid of candidate alphas used in interpolation:
    alphagrid = np.concatenate(
        [np.array([0]),
         10 ** np.arange(np.floor(np.log10(val2)),
                         np.ceil(np.log10(val1)), BIAS_STEP)])

    # The scaling factor applied to coefficients in the rotated space is
    # lambda**2 / (lambda**2 + alpha), where lambda are the singular values
    seltsq = selt**2
    sclg = seltsq / (seltsq + alphagrid[:, None])
    sclg_sq = sclg**2

    # Prellocate the solution:
    if nn >= pp:
        first_dim = pp
    else:
        first_dim = nn

    coef = np.empty((first_dim, ff, bb))
    alphas = np.empty((ff, bb))

    # The main loop is over targets:
    for ii in range(y.shape[-1]):
        # Applies the scaling factors per alpha
        newlen = np.sqrt(sclg_sq @ ols_coef[..., ii]**2).T
        # Normalize to the length of the unregularized solution,
        # because (alphagrid[0] == 0)
        newlen = (newlen / newlen[0])
        # Perform interpolation in a log transformed space (so it behaves
        # nicely), avoiding log of 0.
        temp = interp(fracs, newlen[::-1], np.log(1 + alphagrid)[::-1])
        # Undo the log transform from the previous step
        targetalphas = np.exp(temp) - 1
        # Allocate the alphas for this target:
        alphas[:, ii] = targetalphas
        # Calculate the new scaling factor, based on the interpolated alphas:
        sc = seltsq / (seltsq + targetalphas[np.newaxis].T)
        # Use the scaling factor to calculate coefficients in the rotated
        # space:
        coef[..., ii] = (sc * ols_coef[..., ii]).T

    # After iterating over all targets, we unrotate using the unitary v
    # matrix and reshape to conform to desired output:
        
    pred = np.reshape(X_test @ v_t.T @ coef.reshape((first_dim, ff * bb)),
                  (X_test.shape[0], ff, bb)).squeeze()
    
    if betas_wanted:
        coef = np.reshape(v_t.T @ coef.reshape((first_dim, ff * bb)),
          (pp, ff, bb)).squeeze()
    else:
        coef = None
    
    return pred, coef, alphas



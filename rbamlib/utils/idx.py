
import numpy as np


def idx(arr, val, tol=None):
    r"""
    Get the index of `val` from array `arr` with an optional tolerance.

    If `tol` is specified and the minimum absolute difference between any array
    element and `val` is greater than `tol`, returns NaN. Without `tol`, returns
    the index of the nearest value in `arr` to `val`.

    Parameters
    ----------
    arr : array_like
        Input array.
    val : float
        The value to search for.
    tol : float, optional
        Tolerance for the index selection. If not provided, the nearest index is returned.

    Returns
    -------
    int or NaN
        The index of `val` in `arr` if within `tol` when specified, otherwise the index
        of the nearest value. Returns NaN if the condition is not met or `val` is not found
        within `tol`.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> idx(arr, 3.1)
    2
    >>> idx(arr, 3.1, tol=0.05)
    nan
    """
    # Compute absolute difference and find index of minimum value
    diff = np.abs(arr - val)
    i = np.argmin(diff)

    # If tolerance is specified and the minimum difference exceeds tol, return NaN
    if tol is not None and diff[i] > tol:
        return np.nan
    else:
        return i

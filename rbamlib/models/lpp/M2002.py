import numpy as np


def M2002(time, kp):
    r"""
    Calculates plasmapause location (lpp) following Moldwin et al. (2002) [#]_ eq. (2).

    Parameters
    ----------
    time : ndarray
        Vector of time in days as a numpy array.
    kp : ndarray
        Vector of Kp index, same size as time.

    Returns
    -------
    lpp : ndarray
        Vector of plasmapause location in Re, same size as time.

    Notes
    -----
    The plasmapause location Lpp is calculated using the formula:

    .. math::
        L_{pp} = 5.39 - 0.382 \cdot Kp_{12}

    where \(Kp_{12}\) is the maximum Kp-index value in the last 12 hours.

    Notes
    -----
    .. [#] Moldwin, M. B., et al. (2002). A new model of the location of the plasmapause: CRRES results, Journal of Geophysical Research, 107(A11), 1339, doi: 10.1029/2001JA009211.
    """
    # Ensure time and index arrays are floats
    time = time.astype(float)
    kp = kp.astype(float)

    # Initialize output array for lpp
    lpp = np.zeros_like(time)

    for it, current_time in enumerate(time):
        # Get indices where time is within the last 12 hours (1/2 day)
        tidx = (time > (current_time - 1 / 2)) & (time <= current_time)

        # Get the maximum Kp value in the last 12 hours
        Kp12 = np.max(kp[tidx]) if np.any(tidx) else 0  # Handle empty index case

        # Calculate lpp for the current time step
        lpp[it] = 5.39 - 0.382 * Kp12

    return lpp

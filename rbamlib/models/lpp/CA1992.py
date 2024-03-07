import numpy as np


def CA1992(time, kp):
    r"""
    Calculates plasmapause location Lpp following Carpenter and Anderson (1992) [1]_, eq (7).

    Parameters
    ----------
    time : ndarray
        Time in days as a numpy array.
    kp : ndarray
        Kp-index, vector of geomagnetic activity indices corresponding to the times.

    Returns
    -------
    lpp : ndarray
        Plasmapause location Lpp as a vector.

    Notes
    -----
    The plasmapause location Lpp is calculated using the formula:

    .. math::
        L_{pp} = 5.6 - 0.46 \cdot Kp_{24}

    where \(Kp_{24}\) is the maximum Kp-index value in the last 24 hours.

    References
    ----------
    [1] Carpenter, D. L., & Anderson, R. R. (1992). An ISEE/whistler model of equatorial electron density in the
    magnetosphere. Journal of Geophysical Research, [Space Physics], 97(A2), 1097–1108.
    https://doi.org/10.1029/91JA01548
    """
    left_index = 0
    lpp = np.zeros(len(time))
    for it in range(len(time)):
        # Faster search of the last 24 hours time
        if time[it] - 1 > time[left_index]:
            search_index = np.arange(left_index, it + 1)
            search_time = time[left_index:it + 1]
            i = np.argmin(np.abs(search_time - (time[it] - 1)))
            left_index = search_index[i]
        Kp24 = np.max(kp[left_index:it + 1])
        lpp[it] = 5.6 - 0.46 * Kp24

    return lpp

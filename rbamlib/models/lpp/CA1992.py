import numpy as np


def CA1992(time, kp):
    r"""
    Calculates plasmapause location Lpp following Carpenter and Anderson :cite:yearpar:`carpenter:1992`, eq (7).

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

    where :math:`Kp_{24}` is the maximum Kp-index value in the last 24 hours.
    """

    # Ensure time and index arrays are floats
    time = time.astype(float)
    kp = kp.astype(float)

    left_index = 0
    lpp = np.zeros_like(time)
    for it, current_time in enumerate(time):
        # Faster search of the last 24 hours time
        if current_time - 1 > time[left_index]:
            search_index = np.arange(left_index, it + 1)
            search_time = time[left_index:it + 1]
            i = np.argmin(np.abs(search_time - (current_time - 1)))
            left_index = search_index[i]

        Kp24 = np.max(kp[left_index:it + 1])
        lpp[it] = 5.6 - 0.46 * Kp24

    return lpp

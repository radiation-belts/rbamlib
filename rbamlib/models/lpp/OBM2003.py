import numpy as np


def OBM2003(time, index, index_type):
    r"""
    Calculates plasmapause location (lpp) following O’Brien and Moldwin (2003) [#]_ eq. (1).

    Parameters
    ----------
    time : ndarray
        Time in days as a numpy array.
    index : ndarray
        Vector of the index corresponding to the type, same size as time.
    index_type : str
        A string representing the type of index ('Kp', 'Ae', 'Dst').

    Returns
    -------
    lpp : ndarray
        Vector plasmapause location, same size as time.

    Notes
    -----
    The plasmapause location Lpp is calculated based on the empirical model by O'Brien and Moldwin (2003).

    .. math::
        L_{pp} = a \cdot Q + b

    where \( Q \) is a representation of one of the indicators:

    .. math::
       Q_{Kp} = \max_{-36,-2} Kp, \\
       Q_{AE} = \log_{10} \left( \max_{-36,0} AE \right), \\
       Q_{Dst} = \log_{10} \left| \min_{-24,0} Dst \right|

    When using Kp index, the first value will always be np.nan because maximum Kp is calculated between -36 and -2 hours prior current time

    References
    ----------
    .. [#] O'Brien, T. P., & Moldwin, M. B. (2003). Empirical plasmapause models from magnetic indices. Geophysical Research Letters, 30(4), 1152. https://doi.org/10.1029/2002GL016007
    """

    # Convert the index type to lowercase
    index_type = index_type.lower()

    # Define coefficients and time limits for each index type
    if index_type == 'kp':
        a, b = -0.43, 5.9
        t1, t2 = -36 / 24, -2 / 24  # Convert hours to days
    elif index_type == 'ae':
        a, b = -2.86, 12.4
        t1, t2 = -36 / 24, 0  # AE uses the last 36 hours
    elif index_type == 'dst':
        a, b = -1.57, 6.3
        t1, t2 = -24 / 24, 0  # Dst uses the last 24 hours
    else:
        raise ValueError('Unknown index type')

    # Ensure time and index arrays are floats
    time = time.astype(float)
    index = index.astype(float)

    # Initialize the lpp array with zeros
    lpp = np.zeros_like(time)


    for it, current_time in enumerate(time):
        # Define time window based on t1 and t2
        tidx = (time > (current_time + t1)) & (time <= (current_time + t2))

        # Calculate Q based on the index type
        if np.any(tidx):
            if index_type == 'kp':
                Q = np.max(index[tidx])
            elif index_type == 'ae':
                Q = np.log10(np.max(index[tidx]))
            elif index_type == 'dst':
                Q = np.log10(np.abs(np.min(index[tidx])))
        else:
            Q = None  # If no valid data within the time window

        # Compute Lpp if Q is available
        if Q is not None:
            lpp[it] = a * Q + b
        else:
            lpp[it] = np.nan  # Assign NaN if no valid Q found

    return lpp

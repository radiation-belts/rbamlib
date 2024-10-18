import numpy as np


def OBM2003(time, index, index_type):
    """
    Calculates plasmapause location (lpp) following O’Brien and Moldwin (2003) [1]_ eq. (1).

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
       Q_{Kp} = \max_{-36,-2} Kp, \quad
       Q_{AE} = \log_{10} \left( \max_{-36,0} AE \right)
       Q_{Dst} = \log_{10} \left| \min_{-24,0} Dst \right|

    When using Kp index, the first value will always be np.nan because maximum Kp is calculated between -36 and -2 hours prior current time

    References
    ----------
    [1] O'Brien, T. P., & Moldwin, M. B. (2003). Empirical plasmapause models from magnetic indices.
    Geophysical Research Letters, 30(4), 1152. https://doi.org/10.1029/2002GL016007
    """

    # Convert the index type to lowercase
    index_type = index_type.lower()

    # Set coefficients and time limits based on the index type
    if index_type == 'kp':
        a, b = -0.43, 5.9
        t1, t2 = -36, -2
    elif index_type == 'ae':
        a, b = -2.86, 12.4
        t1, t2 = -36, 0
    elif index_type == 'dst':
        a, b = -1.57, 6.3
        t1, t2 = -24, 0
    else:
        raise ValueError('Unknown index type')

    # Initialize the lpp array with zeros
    lpp = np.zeros(len(time))

    for it in range(len(time)):
        # Define time window based on t1 and t2
        tidx = (time > (time[it] + t1 / 24)) & (time <= (time[it] + t2 / 24))

        # Calculate Q based on the index type
        if index_type == 'kp':
            Q = np.max(index[tidx]) if np.any(tidx) else None
        elif index_type == 'ae':
            Q = np.log10(np.max(index[tidx])) if np.any(tidx) else None
        elif index_type == 'dst':
            Q = np.log10(np.abs(np.min(index[tidx]))) if np.any(tidx) else None

        # Compute Lpp if Q is available
        if Q is not None:
            lpp[it] = a * Q + b
        else:
            lpp[it] = np.nan  # Assign NaN if no valid Q found

    return lpp

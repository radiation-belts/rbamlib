import numpy as np

def mlt2phi(mlt):
    r"""
    Convert Magnetic Local Time (MLT, hours) to magnetic local time angle :math:`\phi` (radians).
    
    Parameters
    ----------
    mlt : float or ndarray
        Magnetic Local Time in hours.

    Returns
    -------
    float or ndarray
        Magnetic local time angle :math:`\phi` in radians.

    Notes
    -----
    - MLT = 12 h → :math:`\phi = 0`  (noon)
    - MLT = 18 h → :math:`\phi = \pi/2`  (dusk)
    - MLT = 0 h  → :math:`\phi = \pi` (midnight)
    - MLT = 6 h  → :math:`\phi = 3\pi/2`  (dawn)

    .. math::
       \phi = \left( \mathrm{MLT} \bmod 24 - 12 \right)\frac{\pi}{12}
        
    Examples
    --------
    >>> mlt2phi(12.0)  # noon
    0.0
    >>> mlt2phi(0.0)   # midnight
    3.141592653589793    
    """
    mlt = np.asarray(mlt, dtype=float)
    phi = ((mlt % 24.0) - 12.0) * (np.pi / 12.0)
    return phi % (2.0 * np.pi)


import numpy as np

def phi2mlt(phi):
    r"""
    Convert magnetic local time angle :math:`\phi` (radians) to Magnetic Local Time (MLT, hours).

    Parameters
    ----------
    phi : float or ndarray
        Magnetic local time angle :math:`\Phi` in radians. Can be any real value; the
        mapping is periodic with :math:`2\pi`.

    Returns
    -------
    float or ndarray
        Magnetic Local Time (MLT) in hours.

    Notes
    -----
    - :math:`\phi = 0`      → MLT = 12 h (noon)
    - :math:`\phi = \pi/2`  → MLT = 18 h (dusk)
    - :math:`\phi = \pi`    → MLT = 0 h  (midnight)
    - :math:`\phi = 3\pi/2` → MLT = 6 h  (dawn)

    .. math::
       {\rm MLT} = \left( \frac{12\,\Phi}{\pi} + 12 \right) \bmod 24.


    Examples
    --------
    >>> import numpy as np
    >>> phi2mlt(0.0)                     # φ = 0 -> MLT = 12 h (noon)
    12.0
    >>> phi2mlt(np.pi)                   # φ = π -> MLT = 24 ≡ 0 h (midnight)
    0.0
    >>> phi2mlt(3*np.pi/2)                # φ = 3π/2 -> MLT = 18 h
    18.0
    """
    phi = np.asarray(phi, dtype=float)
    return (12.0 * phi / np.pi + 12.0) % 24.0

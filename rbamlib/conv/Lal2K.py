import numpy as np
import rbamlib.models.dip


def Lal2K(L, al=np.pi/2, B0=rbamlib.models.dip.B0):
    r"""
    Calculate the adiabatic invariant K, from of L-shell and pitch-angle.


    Parameters
    ----------
    L : float or ndarray
        L-shell value
    al : float or ndarray
         Pitch-angle, in radians. Default is 90 degrees, which results in K=0.
    B0 : function, default=rbamlib.models.dip.B0
        Function that calculates the magnetic field in Gauss at a given L-shell `L` (or `r`). By default, Earth's dipole field model is used.

    Returns
    -------
    float or ndarray
         The adiabatic invariant K, in units of G^0.5 * R

    Notes
    -----
    .. math::
        K = \frac{Y(\alpha)}{\sin(\alpha)} \cdot L \cdot \sqrt{B_0(L)}

    where:
        - :math:`\alpha` is the pitch-angle in radians
        - :math:`L` is the L-shell
        - :math:`Y(\alpha)` is a Y-function
        - :math:`B_0(L)` is the magnetic field strength at L-shell L.

    See Also
    --------
    rbamlib.models.dip.Y : Y-function
    rbamlib.models.dip.B0 : Dipole magnetic field

    """
    b = B0(L)
    y = rbamlib.models.dip.Y(al)
    K = y / np.sin(al) * L * np.sqrt(b)
    return K

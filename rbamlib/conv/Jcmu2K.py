import numpy as np
import rbamlib.constants


def Jcmu2K(Jc, mu):
    r"""
    Calculates adiabatic invariant K.

    Calculates the adiabatic invariant K from the second adiabatic invariant Jc
    and the first adiabatic invariant mu.

    Parameters
    ----------
    Jc : float or ndarray
        The second adiabatic invariant times the speed of light, Jc = J*c,
        where J is the second adiabatic and c is the speed of light. Units are MeV/G.
    mu : float or ndarray
        The first adiabatic invariant mu, in MeV/Gauss

    Returns
    -------
    float or ndarray
        The adiabatic invariant K, in units of G^0.5 * R

    Notes
    -----
    .. math::
        K = \frac{Jc}{\sqrt{8 \cdot mc^2 \cdot \mu}}

    where :math:`mc^2` (the rest mass energy of an electron) is 0.511 MeV.
    """

    mc2 = rbamlib.constants.MC2
    K = Jc / np.sqrt(8 * mc2 * mu)
    return K

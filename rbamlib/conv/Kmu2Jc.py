import numpy as np
import rbamlib.constants


def Kmu2Jc(K, mu):
    r"""
    Calculates adiabatic invariant Jc = J*c

    Calculates the second adiabatic invariant Jc from the adiabatic invariant K and the first adiabatic invariant mu.

    Parameters
    ----------
    K : float or ndarray
        The adiabatic invariant K, in Gauss^0.5 * Re.
    mu : float or ndarray
        The first adiabatic invariant mu, in MeV/Gauss

    Returns
    -------
    float or ndarray
        The second adiabatic invariant times the speed of light, Jc = J*c,
        where J is the second adiabatic and c is the speed of light. Units are MeV/Gauss.

    Notes
    -----
    .. math::
        Jc = K \\cdot \\sqrt{8 \\cdot mc^2 \\cdot \\mu}

    where :math:`mc^2` (the rest mass energy of an electron) is 0.511 MeV.
    """
    mc2 = rbamlib.constants.MC2
    Jc = K * np.sqrt(8 * mc2 * mu)
    return Jc

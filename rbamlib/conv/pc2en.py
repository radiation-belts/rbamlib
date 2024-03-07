import numpy as np
import rbamlib.constants


def pc2en(pc):
    r"""
    Calculates energy (MeV) from p*c (MeV).

    Parameters
    ----------
    pc : float or ndarray
        The momentum times the speed of light (p*c) in MeV, either as a single floating-point number or an ndarray of numbers.

    Returns
    -------
    en : ndarray
        The energy in MeV corresponding to the given momentum p*c.

    Notes
    -----
    .. math::
        en = \left( \sqrt{1 + \left(\frac{pc}{mc^2}\right)^2} - 1 \right) \cdot mc^2

    where :math:`mc^2` (the rest mass energy of an electron) is 0.511 MeV.
    """
    mc2 = rbamlib.constants.MC2
    en = (np.sqrt(1 + (pc / mc2) ** 2) - 1) * mc2
    return en

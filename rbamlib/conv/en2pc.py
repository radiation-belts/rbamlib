import numpy as np
import rbamlib.constants


def en2pc(en):
    r"""
    Calculates p*c (MeV) from energy (MeV).

    Parameters
    ----------
    en : float or ndarray
        The energy or an array of energies in MeV.

    Returns
    -------
    pc : ndarray
        The momentum times the speed of light (p*c) in MeV corresponding to the input energy.

    Notes
    -----
    .. math::
        pc = \sqrt{\left(\frac{energy}{mc^2} + 1\right)^2 - 1} \cdot mc^2

    where :math:`mc^2` (the rest mass energy of an electron) is 0.511 MeV.
    """

    mc2 = rbamlib.constants.MC2
    pc = np.sqrt((en / mc2 + 1) ** 2 - 1) * mc2
    return pc

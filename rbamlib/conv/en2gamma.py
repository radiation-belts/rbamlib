import numpy as np
import rbamlib.constants


def en2gamma(en, mc2=rbamlib.constants.MC2):
    r"""
    Calculates the Lorentz factor (γ) from kinetic energy (MeV).

    Parameters
    ----------
    en : float or ndarray
        The kinetic energy or an array of kinetic energies in MeV.
    mc2 : float, optional
        The rest mass energy (m*c^2) in MeV. Default is provided for electrons (typically 0.511 MeV).

    Returns
    -------
    gamma : float or ndarray
        The Lorentz factor corresponding to the input kinetic energy.

    Notes
    -----
    The total energy of a particle is given by

    .. math::
        E = \gamma m c^2

    where the Lorentz factor is

    .. math::
        \gamma = \frac{E}{mc^2} = \frac{K}{mc^2} + 1

    Here, K is the kinetic energy.
    """
    gamma = en / mc2 + 1.0
    return gamma
import numpy as np
from rbamlib.conv import mural2pc, pc2en


def mural2en(mu, r, al=np.pi / 2, *B0):
    r"""
    Convert first adiabatic invariant, mu, to energy in MeV

    Parameters
    ----------
    mu : float or ndarray
        First adiabatic invariant, mu, in MeV/G.
    r : float or ndarray
        Radial distance, expressed in the planet radii (R).
    al : float or ndarray, default=np.pi / 2
        Pitch-angle, in radians. Default is 90 degrees.
    *B0 : function, optional
        Argument of `mural2pc` defining  function that calculates the magnetic field

    Returns
    -------
    float or ndarray
        Energy, in MeV.

    Notes
    -----
    This function is a wrapper of `mural2pc` and `pc2en`.

    See Also
    --------
    mural2pc : Calculate the momentum times the speed of light (pc), in MeV from mu.
    pc2en : Calculates energy (MeV) from p*c (MeV).
    """

    # Pass any additional arguments *B0 directly to mural2pc
    pc = mural2pc(mu, r, al, *B0)
    en = pc2en(pc)
    return en


# Alias
def mu2en(mu, r, al=np.pi / 2, *B0):
    r"""
    Convert first adiabatic invariant, mu, to energy in MeV

    See Also
    --------
    mural2en: Alias of the `mural2en` function.
    """

    return mural2en(mu, r, al, *B0)

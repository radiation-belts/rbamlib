import numpy as np
from rbamlib.conv import pcral2mu, en2pc


def enral2mu(en, r, al=np.pi/2, *B0):
    r"""
    Convert energy in MeV to first adiabatic invariant, mu, in MeV/G.

    Parameters
    ----------
    en : float or ndarray
        Energy, in MeV.
    r : float or ndarray
        Radial distance, expressed in the planet radii (R).
    al : float or ndarray, default=np.pi / 2
        Pitch-angle, in radians. Default is 90 degrees.
    *B0 : function, optional
        Argument of `pcral2mu` defining function that calculates the magnetic field

    Returns
    -------
    float or ndarray
        First adiabatic invariant, mu, in MeV/G.

    Notes
    -----
    This function is a wrapper of `pcral2mu` and `en2pc`.

    See Also
    --------
    pcral2mu :  Calculate first adiabatic invariant mu from pc.
    en2pc : Calculates p*c (MeV) from energy (MeV).
    """

    # Pass any additional arguments *B0 directly to pcral2mu
    pc = en2pc(en)
    mu = pcral2mu(pc, r, al, *B0)
    return mu


# Alias
def en2mu(en, r, al=np.pi/2, *B0):
    r"""
    Convert energy in MeV to first adiabatic invariant, mu, in MeV/G.

    See Also
    --------
    enral2mu: Alias of the `enral2mu` function.
    """

    return enral2mu(en, r, al, *B0)

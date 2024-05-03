import numpy as np
import rbamlib.constants
import rbamlib.models.dip


def mural2pc(mu, r, al=np.pi / 2, B0=rbamlib.models.dip.B0):
    r"""
    Calculate the momentum times the speed of light (pc), in MeV.

    Parameters
    ----------
    mu : float or ndarray
        First adiabatic invariant, mu, in MeV/G.
    r : float or ndarray
        Radial distance, expressed in the planet radii (R).
    al : float or ndarray, default=np.pi / 2
        Pitch-angle, in radians. Default is 90 degrees.
    B0 : function, default=rbamlib.models.dip.B0
        Function that calculates the magnetic field in Gauss at a given radial distance `r`. By default, Earth's dipole field model is used.

    Returns
    -------
    float or ndarray
        The momentum times the speed of light, pc = p*c, in MeV.

    Notes
    -----
    .. math::
        pc = \frac{\sqrt{2 \cdot \mu \cdot mc^2 \cdot B_0(r)}}{\sin(\alpha)}

    Where:
        - :math:`\mu` is the magnetic moment
        - :math:`mc^2` is the electron rest mass energy equivalent
        - :math:`B_0(r)` is the magnetic field strength in Gauss at distance `r` for the specified `planet`
        - :math:`\alpha` is the pitch-angle

    See Also
    --------
    mu2pc : Alias to the `mural2pc` function.
    """
    mc2 = rbamlib.constants.MC2
    b = B0(r)
    pc = np.sqrt(2.0 * mu * mc2 * b) / np.sin(al)
    return pc


# Alias
def mu2pc(mu, r, al=np.pi / 2, B0=rbamlib.models.dip.B0):
    r"""
    Calculate the momentum times the speed of light (pc), in MeV.
    
    Alias of the `mural2pc` function.
    """

    return mural2pc(mu, r, al, B0)

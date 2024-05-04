import numpy as np
import rbamlib.constants
import rbamlib.models.dip


def pcral2mu(pc, r, al=np.pi/2, B0=rbamlib.models.dip.B0):
    r"""
    Calculate first adiabatic invariant mu from pc (momentum times the speed of light).

    Parameters
    ----------
    pc : float or ndarray
        The momentum times the speed of light, pc = p*c, in MeV.
    r : float or ndarray
        Radial distance, expressed in the planet radii (R).
    al : float or ndarray, default=np.pi / 2
        Pitch-angle, in radians. Default is 90 degrees.
    B0 : function, default=rbamlib.models.dip.B0
        Function that calculates the magnetic field in Gauss at a given radial distance `r`. By default, Earth's dipole field model is used.

    Returns
    -------
    float or ndarray
        First adiabatic invariant, mu, in MeV/G.

    Notes
    -----
    .. math::
        \mu = \frac{{pc^2 \cdot \sin^2(\alpha)}}{{2 \cdot mc^2 \cdot B}}

    Where:
        - :math:`pc` the momentum times the speed of light, pc = p*c,
        - :math:`mc^2` is the electron rest mass energy equivalent
        - :math:`B_0(r)` is the magnetic field strength in Gauss at distance `r` for the specified `planet`
        - :math:`\alpha` is the pitch-angle

    See Also
    --------
    pc2mu : Alias to the `pcral2mu` function.
    rbamlib.models.dip.B0 : Dipole magnetic field
    """
    mc2 = rbamlib.constants.MC2
    b = B0(r)
    mu = pc ** 2 * np.sin(al) ** 2 / (b * 2 * mc2)
    return mu


# Alias
def pc2mu(pc, r, al=np.pi/2, *B0):
    r"""
    Calculate first adiabatic invariant mu from pc (momentum times the speed of light).

    See Also
    --------
    pcral2mu: Alias of the `pcral2mu` function.
    """

    return pcral2mu(pc, r, al, *B0)

import numpy as np
import rbamlib.constants


def Y(al):
    """
    Approximation of the integral function Y related to the second adiabatic invariant, derived in the dipole approximation.

    Parameters
    ----------
    al : float or ndarray
         Equatorial pitch angle, in radians.

    Returns
    -------
    float or ndarray
         Value of Y

    Notes
    -----
        See Schulz & Lanzerotti (1974) [#]_.

    .. math::
        Y( \\alpha ) \\approx 2(1 - \\sin( \\alpha ))T_0 + (T_0 - T_1) \\cdot \\left( \\sin( \\alpha ) \\cdot \\ln( \\sin( \\alpha ) ) + 2 \\sin( \\alpha ) - 2 \\sqrt{ \\sin( \\alpha ) } \\right)

    References
    ----------
    .. [#] Schulz, M., & Lanzerotti, L. J. (1974). Particle Diffusion in the Radiation Belts (Vol. 7). Springer-Verlag Berlin Heidelberg. Retrieved from http://www.springer.com/physics/book/978-3-642-65677-4
    """
    T0 = rbamlib.constants.T0
    T1 = rbamlib.constants.T1

    # Since np.log(y) is -inf in case of al = 0, we return/change to np.nan
    if np.all(al == 0):
        return np.nan
    elif np.any(al == 0):
        al[al == 0] = np.nan

    y = np.sin(al)

    Y = 2 * (1 - y) * T0 + (T0 - T1) * (y * np.log(y) + 2. * y - 2. * np.sqrt(y))
    return Y

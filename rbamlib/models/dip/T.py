import numpy as np
import rbamlib.constants


def T(al):
    """
    Approximation of the integral function T related to the bounce period, derived in the dipole approximation.

    Parameters
    ----------
    al : float or ndarray
         Equatorial pitch angle, in radians.

    Returns
    -------
    float or ndarray
         Value of T

    Notes
    -----
        See Schulz, M., & Lanzerotti, L. J. (1974). Particle Diffusion in the Radiation Belts (Vol. 7). Springer-Verlag Berlin Heidelberg. Retrieved from http://www.springer.com/physics/book/978-3-642-65677-4

    .. math::
        T( \\alpha ) \\approx T_0 - \\frac{1}{2}(T_0 - T_1) \\cdot \\left( \\sin( \\alpha ) + \\sin( \\alpha)^1/2 \\right)
    """
    T0 = rbamlib.constants.T0
    T1 = rbamlib.constants.T1

    y = np.sin(al)

    T = T0 - 0.5 * (T0 - T1) * (y + np.sqrt(y))
    return T

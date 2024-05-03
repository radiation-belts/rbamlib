import numpy as np
import rbamlib.constants


def B(r, mlat=0, planet='Earth'):
    """
    The dipole model of magnetic field.

    Parameters
    ----------
    r : float or ndarray
         The distance from the center of the planet, in planet's radii.

    mlat : float or ndarray, default = 0
         The magnetic latitude (MLAT), or geomagnetic latitude, is measured northwards from the equator in radians

    planet : str, default = 'Earth'
         The planet.

    Returns
    -------
    float or ndarray
         The magnetic field strength, in Gauss.
    """

    if planet == 'Earth':
        B0 = rbamlib.constants.B0_Earth
    else:  # Default is Earth
        B0 = rbamlib.constants.B0_Earth

    # If mlat is 0 the calculation can be simplified
    if np.all(mlat == 0):
        return rbamlib.models.dip.B0(r, planet)

    B = B0 / r ** 3 * np.sqrt(1 + 3 * np.sin(mlat) ** 2)
    return B

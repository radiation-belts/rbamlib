import rbamlib.constants


def B0(r, planet='Earth'):
    """
     Magnetic field value at the magnetic equator.

     Simplified calculation of the magnetic field at the magnetic equator.

    Parameters
    ----------
    r : float or ndarray
         The distance from the center of the planet, in planet's radii.

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

    B = B0 / r ** 3
    return B

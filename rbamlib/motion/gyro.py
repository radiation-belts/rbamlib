import numpy as np
import rbamlib.constants
from rbamlib.conv import en2gamma
from rbamlib.models.dip import B0

def f_gyro(B=None, L=None, en=None, planet='Earth', q=rbamlib.constants.q, m=rbamlib.constants.me):
    r"""
    Calculate the oscillation gyro (cyclotron) frequency for a charged particle.
    
    It accounts for the relativistic correction when the particle's kinetic energy is provided.

    Parameters
    ----------
    B : float or ndarray, optional
        Magnetic field strength, in Gauss. Either B or L must be provided.
    L : float or ndarray, optional
        McIlwain L-shell parameter (dimensionless). Used to compute B if it is not provided.
    en : float or ndarray, optional
        Kinetic energy of the particle, in MeV. When provided, a relativistic correction via 
        the Lorentz factor is applied.
    planet : str, optional, default = 'Earth'
        Name of the planet for which the magnetic field is calculated.
    q : float, optional
        Particle charge, in statcoulombs (SGS units). Default is electrons.
    m : float, optional
        Particle mass, in grams. Default is electrons.

    Returns
    -------
    float or ndarray
        Gyro frequency, in Hertz (Hz).

    Notes
    -----
    For a non-relativistic particle, the gyro frequency is given by

    .. math::
        f = \frac{|q| B}{2 \pi m c}

    For a relativistic particle with kinetic energy (en), the Lorentz factor is

    .. math::
        \gamma = 1 + \frac{en}{m \cdot c^2}

    and the gyro frequency becomes [#]_

    .. math::
       f = \frac{|q| B}{2 \pi \gamma m c}


    References
    ----------
    .. [#] Roederer, J. G., & Zhang, H. (2016). Dynamics of magnetically trapped particles. Berlin, Germany: Springer.

    """
    # Compute gamma if energy is provided (en is in MeV)
    gamma = 1.0
    if en is not None:
        gamma = en2gamma(en)
    
    # Calculate non default option for particles other than electrons
    if m != rbamlib.constants.me:
        erg_to_MeV = 1e-7 / 1.60218e-13  # ≈ 6.24151e5 MeV per erg
        mc2_erg = m * rbamlib.constants.c**2
        # Calculate mc2 in MeV
        mc2 = mc2_erg * erg_to_MeV
        gamma = en2gamma(en, mc2)

    if B is None and L is not None:
        B = B0(L, planet=planet)
    elif B is None and L is None:
        raise KeyError("Either magnetic field (B) or McIlwain L should be provided.")



    # Compute f
    f = np.abs(q) * B / (2 * np.pi * gamma * m * rbamlib.constants.c)

    return f       


def omega_gyro(B=None, L=None, en=None, planet='Earth',  q=rbamlib.constants.q, m=rbamlib.constants.me):
    r"""
    Calculate the angular gyro (cyclotron) frequency for a charged particle in a magnetic field.

    Parameters
    ----------
    B : float or ndarray, optional
        Magnetic field strength, in Gauss. Either B or L must be provided.
    L : float or ndarray, optional
        McIlwain L-shell parameter (dimensionless). Used to compute B if it is not provided.
    en : float or ndarray, optional
        Kinetic energy of the particle, in MeV. When provided, a relativistic correction via 
        the Lorentz factor is applied.
    planet : str, optional, default = 'Earth'
        Name of the planet for which the magnetic field is calculated.
    q : float, optional
        Particle charge, in statcoulombs (SGS units). Defaults is for electrons.
    m : float, optional
        Particle mass, in grams. Defaults is for electrons.

    Returns
    -------
    float or ndarray
        Angular gyro frequency, in radians per second (rad/s).

    Notes
    -----
    The angular gyro frequency is calculated from the standard gyro frequency f [#]_:

    .. math::
        \omega = 2 \pi f

    References
    ----------
    .. [#] Koskinen, H. E. J., & Kilpua, E. K. J. (2022). Physics of earth’s radiation belts. Cham, Switzerland: Springer Nature. https://doi.org/10.1007/978-3-030-82167-8

    """
    return 2 * np.pi * f_gyro(B=B, en=en, L=L, planet=planet, q=q, m=m)


def T_gyro(B=None, L=None, en=None, planet='Earth',  q=rbamlib.constants.q, m=rbamlib.constants.me):
    r"""
    Calculate the gyro period of a charged particle in a magnetic field.

    Parameters
    ----------
    B : float or ndarray, optional
        Magnetic field strength, in Gauss. Either B or L must be provided.
    L : float or ndarray, optional
        McIlwain L-shell parameter (dimensionless). Used to compute B if it is not provided.
    en : float or ndarray, optional
        Kinetic energy of the particle, in MeV. When provided, a relativistic correction via 
        the Lorentz factor is applied.
    planet : str, optional, default = 'Earth'
        Name of the planet for which the magnetic field is calculated.
    q : float, optional
        Particle charge, in statcoulombs (SGS units). Defaults is for electrons.
    m : float, optional
        Particle mass, in grams. Defaults is for electrons.
    
    
    Returns
    -------
    float or ndarray
        Gyro period, in seconds (s).

    Notes
    -----
    The gyro period is defined as the inverse of the angular gyro frequency [#]_:

    .. math::
        T = \frac{2\pi}{\omega}

    References
    ----------
    .. [#] Koskinen, H. E. J., & Kilpua, E. K. J. (2022). Physics of earth’s radiation belts. Cham, Switzerland: Springer Nature. https://doi.org/10.1007/978-3-030-82167-8
    """
    return 2 * np.pi / omega_gyro(B=B, en=en, L=L, planet=planet, q=q, m=m)


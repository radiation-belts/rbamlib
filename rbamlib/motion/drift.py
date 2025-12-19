import rbamlib.models.dip as dip
import rbamlib.constants
from rbamlib.conv import en2gamma
import numpy as np


def f_drift(L, en, al=np.pi/2, planet='Earth', m=rbamlib.constants.me, q=rbamlib.constants.q):
    r"""
    Calculate the drift frequency.

    Parameters
    ----------
    L : float or ndarray
        L-shell parameter.
    en : float or ndarray
        Particle kinetic energy in MeV.
    al : float or ndarray, optional, default=pi/2
        Pitch angle in radians. 
    planet : str, optional, default='Earth'
        Name of the planet. 
    m : float, optional
        Particle mass in grams. Default is electrons.
    q : float, optional
        Particle charge in statcoulombs (CGS units). Default is electrons.

    Returns
    -------
    float or ndarray
        Drift frequency in Hertz (Hz).

    Notes
    -----
    The drift frequency is given by the combined equation [#]_:

    .. math::
       f_{drift} = \left[ \frac{L}{2\pi} \cdot \frac{3 m c^3}{|q| B_0 r_0^2} \cdot \frac{\gamma^2 - 1}{\gamma} \right] \cdot \frac{6 - \frac{Y(\alpha)}{T(\alpha)}}{12}
    
    - :math:`\gamma` is the relativistic Lorentz factor,
    - :math:`B_0` is the magnetic field for the specified planet,
    - :math:`r_0` is the planetary radius,
    - :math:`Y(\alpha)` and  :math:`T(\alpha)` are functions from the dipole model.

    References
    ----------
    .. [#] Schulz, M., & Lanzerotti, L. J. (1974). Particle Diffusion in the Radiation Belts.
           Springer-Verlag Berlin Heidelberg.
    """

    # Compute gamma if energy is provided (en is in MeV)
    gamma = en2gamma(en)

    # Calculate non default option for particles other than electrons
    if m != rbamlib.constants.me:
        erg_to_MeV = 1e-7 / 1.60218e-13  # ≈ 6.24151e5 MeV per erg
        mc2_erg = m * rbamlib.constants.c**2
        # Calculate mc2 in MeV
        mc2 = mc2_erg * erg_to_MeV
        gamma = en2gamma(en, mc2)


    if planet == 'Earth':
        r0 = rbamlib.constants.R_Earth
    elif planet == 'Jupiter':
        r0 = rbamlib.constants.R_Jupiter
    elif planet == 'Saturn':
        r0 = rbamlib.constants.R_Saturn
    else:  # Default is Earth
        r0 = rbamlib.constants.R_Earth

    c = rbamlib.constants.c
    
    factor = 3 * m * c**3 / (np.abs(q) * dip.B0(1, planet) * r0**2)
    f =  L * factor * ( (gamma**2 - 1) / gamma ) / (2 * np.pi)
    g = (6 - dip.Y(al)/dip.T(al)) / 12
    return f * g
    

def omega_drift(L, en, al=np.pi/2, planet='Earth', m=rbamlib.constants.me, q=rbamlib.constants.q):
    r"""
    Calculate the angular drift frequency.

    Parameters
    ----------
    L : float or ndarray
        L-shell parameter.
    en : float or ndarray
        Particle kinetic energy in MeV.
    al : float or ndarray, optional, default=pi/2
        Pitch angle in radians. 
    planet : str, optional, default='Earth'
        Name of the planet. 
    m : float, optional
        Particle mass in grams. Default is electrons.
    q : float, optional
        Particle charge in statcoulombs (CGS units). Default is electrons.

    Returns
    -------
    float or ndarray
        Angular drift frequency in radians per second (rad/s).

    Notes
    -----
    The angular drift frequency is calculated using [#]_:

    .. math::
       \omega_{drift} = 2\pi f_{drift}

    References
    ----------
    .. [#] Schulz, M., & Lanzerotti, L. J. (1974). Particle Diffusion in the Radiation Belts.
           Springer-Verlag Berlin Heidelberg.
    """
    return  2 * np.pi * f_drift(L, en, al, planet, m, q)


def T_drift(L, en, al=np.pi/2, planet='Earth', m=rbamlib.constants.me, q=rbamlib.constants.q):
    r"""
    Calculate the drift period.

    Parameters
    ----------
    L : float or ndarray
        L-shell parameter.
    en : float or ndarray
        Particle kinetic energy in MeV.
    al : float or ndarray, optional, default=pi/2
        Pitch angle in radians. 
    planet : str, optional, default='Earth'
        Name of the planet. 
    m : float, optional
        Particle mass in grams. Default is electrons.
    q : float, optional
        Particle charge in statcoulombs (CGS units). Default is electrons.

    Returns
    -------
    float or ndarray
        Drift period in seconds (s).

    Notes
    -----
    The drift period is the reciprocal of the drift frequency [#]_:

    .. math::
       T_{drift} = \frac{1}{f_{drift}}

    References
    ----------
    .. [#] Schulz, M., & Lanzerotti, L. J. (1974). Particle Diffusion in the Radiation Belts.
           Springer-Verlag Berlin Heidelberg.
    """
    return 1 / f_drift(L, en, al, planet, m, q)


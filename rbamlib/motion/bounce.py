import rbamlib.models.dip as dip
import rbamlib.constants
import numpy as np

def f_bounce(L, al, en, planet='Earth', m=rbamlib.constants.me):
    r"""
    Calculate the bounce frequency of a charged particle in a dipolar magnetic field.

    Parameters
    ----------
    L : float or ndarray
        L-shell parameter (dimensionless).
    al : float or ndarray
        Equatorial pitch angle of the particle, in radians.
    en : float or ndarray
        Kinetic energy of the particle, in MeV.
    planet : str, optional, Default = 'Earth'.
        Name of the planet.        
    m : float, optional
        Particle mass, in grams. Default is for electrons.

    Returns
    -------
    float or ndarray
        Bounce frequency, in Hertz (Hz).

    Notes
    -----
    The bounce frequency is defined as the inverse of the bounce period. 

        .. math::
        f_{\mathrm{bounce}} = \\frac{1}{T_{\\mathrm{bounce}}}

    References
    ----------
    .. [#] Schulz, M., & Lanzerotti, L. J. (1974). Particle Diffusion in the Radiation Belts.
           Springer-Verlag Berlin Heidelberg.
    """

    T = T_bounce(L, al, en, planet, m)
    return 1 / T

def omega_bounce(L, al, en, planet='Earth', m=rbamlib.constants.me):
    r"""
    Calculate the angular bounce frequency for a trapped charged particle in a dipolar magnetic field.

    Parameters
    ----------
    L : float or ndarray
        L-shell parameter (dimensionless).
    al : float or ndarray
        Equatorial pitch angle of the particle, in radians.
    en : float or ndarray
        Kinetic energy of the particle, in MeV.
    planet : str, optional, Default = 'Earth'.
        Name of the planet.        
    m : float, optional
        Particle mass, in grams. Default is for electrons.

    Returns
    -------
    float or ndarray
        Angular bounce frequency, in radians per second (rad/s).

    Notes
    -----
    The angular bounce frequency is obtained by:

    .. math::
        \\omega_{\\mathrm{bounce}} = \\frac{2\\pi}{T_{\\mathrm{bounce}}}

    where T is the bounce period calculated by T_bounce.

    References
    ----------
    .. [#] Schulz, M., & Lanzerotti, L. J. (1974). Particle Diffusion in the Radiation Belts.
           Springer-Verlag Berlin Heidelberg.
    """
    T = T_bounce(L, al, en, planet, m)
    return 2 * np.pi / T


def T_bounce(L, al, en, planet='Earth', m=rbamlib.constants.me):
    r"""
    Calculate the bounce period of a trapped charged particle in a dipolar magnetic field.

    Parameters
    ----------
    L : float or ndarray
        L-shell parameter (dimensionless).
    al : float or ndarray
        Equatorial pitch angle of the particle, in radians.
    en : float or ndarray
        Kinetic energy of the particle, in MeV.
    planet : str, optional, Default = 'Earth'.
        Name of the planet.        
    m : float, optional
        Particle mass, in grams. Default is for electrons.

    Returns
    -------
    float or ndarray
        Bounce period, in seconds (s).

    Notes
    -----
    See Schulz & Lanzerotti (1974) [#]_.
    The bounce period is calculated using the following scaling:
    
        T_{bounce} = \\left( \\frac{4r}{v} \\right) T(\\alpha)

    where:
      - r = L * r₀ with r₀ being the planetary radius,
      - v is the effective particle velocity computed using the conversion of energy to momentum,
      - T(\\alpha) is a modulation factor dependent on the equatorial pitch angle 
        provided by rbamlib.models.dip.T(al).

    The velocity v is computed as:

        v = \\frac{pc}{m\\,\\gamma}

    with pc (momentum times speed of light) obtained from the energy conversion and 
    \$ \\gamma \$ being the Lorentz factor.

    Math Expressions
    ----------------
    .. math::
        r = L \\cdot r_0

    .. math::
        v = \\frac{pc}{m\\,\\gamma}, \\quad \\text{with} \\quad pc = \\text{en2pc}(en) \\times \\text{MeV-to-erg}

    .. math::
        T_{\\mathrm{bounce}} = \\frac{4r}{v} \\, T(\\alpha)

    References
    ----------
    .. [#] Schulz, M., & Lanzerotti, L. J. (1974). Particle Diffusion in the Radiation Belts.
           Springer-Verlag Berlin Heidelberg.
    """
    T = dip.T(al)


    if planet == 'Earth':
        r0 = rbamlib.constants.R_Earth
    elif planet == 'Jupiter':
        r0 = rbamlib.constants.R_Jupiter
    elif planet == 'Saturn':
        r0 = rbamlib.constants.R_Saturn
    else:  # Default is Earth
        r0 = rbamlib.constants.R_Earth

    r = L * r0    
    MeV_to_erg = 1.60218e-6 # ≈ 6.24151e5 MeV per erg
    pc = rbamlib.conv.en2pc(en) * MeV_to_erg

    v = (pc / rbamlib.constants.c) / (m * rbamlib.conv.en2gamma(en))
   
    T_bounce = (4 * r / v) * T

    return T_bounce
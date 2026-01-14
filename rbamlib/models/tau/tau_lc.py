import numpy as np
import rbamlib.constants
import rbamlib.models.dip as dip
from rbamlib.motion.bounce import T_bounce

def tau_lc(L, al, en, planet='Earth', m=rbamlib.constants.me, al_lc=None):
    r"""
    Calculate the characteristic lifetime (`tau_lc`) of a charged particle in a dipolar magnetic field.
    
    This lifetime is defined only for particles with equatorial pitch angle strictly
    inside the loss cone (α < α_lc). For α ≥ α_lc, the function returns NaN.
    
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
        Particle mass, in grams. Default is electrons.
    al_lc : float or ndarray, optional
        Equatorial loss-cone angle α_lc (radians). If not
        provided, it is computed with :func:`rbamlib.models.dip.al_lc`.

    Returns
    -------
    float or ndarray
        Characteristic lifetime (`tau_lc`) in seconds. Returns NaN for al ≥ al_lc.

    Notes
    -----

    The lifetime is calculates as quarter of the the bounce period:
    
    .. math::
        \tau_{lc} = \frac{T_{bounce}}{4},

    where :math:`T_{bounce}` is computed by :func:`rbamlib.motion.bounce.T_bounce`.
    This approximation follows Schulz and Lanzerotti (1974) [#]_ and accounts for relativistic effects.


    References
    ----------
    .. [#] Schulz, M., & Lanzerotti, L. J. (1974). Particle Diffusion in the Radiation Belts.
           Springer-Verlag Berlin Heidelberg.
    """

    # Determine the loss-cone angle if not supplied
    al_lc_val = dip.al_lc(L) if al_lc is None else al_lc

    # Broadcast shapes and compute bounce period
    T_bounce_seconds = T_bounce(L, al, en, planet, m)

    # Valid only for alpha strictly inside the loss cone
    inside = np.array(al) < np.array(al_lc_val)

    # Return tau = T_bounce / 4 for valid inputs; NaN otherwise
    tau = np.where(inside, T_bounce_seconds / 4.0, np.nan)

    # Preserve scalar type when inputs are scalar
    if np.isscalar(L) and np.isscalar(al) and np.isscalar(en) and np.isscalar(al_lc_val):
        return float(tau)
    
    return tau

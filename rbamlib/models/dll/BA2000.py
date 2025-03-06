from rbamlib import constants
from rbamlib.models import dip

def BA2000(L, kp, mu=None, dll_type='M'):
    r"""
    Calculate electromagnetic and electrostatic radial diffusion coefficient following Brautigam & Albert (2000) [#]_ model, eq. (6) and eq. (4, 5).

    Parameters
    ----------
    kp : ndarray
        Kp-index, vector of geomagnetic activity indices corresponding to the times.
    L : ndarray
        Array of L values, representing the radial distance in Earth radii where magnetic field lines cross the magnetic equator.
    mu : ndarray, default=None
        Array of mu values, vector of corresponding first adiabatic invariants required for calculation electrostatic radial diffusion coefficient, in MeV/G.
        mu is not used in calculation of electromagnetic radial diffusion coefficient.
    dll_type : str, default='M'
        A string, indicating which diffusion coefficient to calculate.
        'M' stands for electromagnetic radial diffusion coefficient (eq. 6).
        'E' stands for electrostatic diffusion coefficient (eq. 4, 5).
        'ME', 'both', or '' computes both coefficients and returns a tuple (dllm, dlle).

    Returns
    -------
    dllm : numpy.ndarray
        Electromagnetic radial diffusion coefficient, in 1/days (if dll_type='M').
    dlle : numpy.ndarray
        Electrostatic radial diffusion coefficient, in 1/days (if dll_type='E').
    dllm, dlle: tuple of numpy.ndarray
        Both electromagnetic and electrostatic diffusion coefficients (dllm, dlle) if dll_type='ME' or ''.

    Notes
    -----
    The electromagnetic radial diffusion coefficient is calculated as:

    .. math::
        D^{M}_{LL} = 10^{(0.506 \cdot Kp - 9.325)} \cdot L^{10}

    The electrostatic radial diffusion coefficient is calculated as:

    .. math::
        D^{E}_{LL} = \frac{1}{4} \left(\frac{c E_{ms}}{B_0}\right)^2 \frac{T}{1 + (\omega_D T / 2)^2} L^6

    where:

    .. math::
        \omega_D = \frac{3 \mu c}{e L^2 R_E^2} \left(1 + \frac{2 \mu B_}{E_0}\right)^{-1/2}

    .. math::
        E_{ms} = 0.26 (Kp - 1) + 0.1

    References
    ----------
    .. [#] Brautigam, D. H., & Albert, J. M. (2000). Radial diffusion analysis of outer radiation belt electrons during the October 9, 1990, magnetic storm. Journal of Geophysical Research, 105(A1), 291–309. https://doi.org/10.1029/1999ja900344
    """

    dllm, dlle = None, None

    if dll_type in {'M', 'm', '', 'ME', 'me', 'both'}:
        # Electromagnetic diffusion coefficient
        dllm = 10 ** (0.506 * kp - 9.325) * L ** 10

    if dll_type in {'E', 'e', '', 'ME', 'me', 'both'}:
        if mu is None:
            raise ValueError("'mu' is required for calculating electrostatic diffusion (dll_type='E').")

        # Constants
        B0 = constants.B0_Earth
        B0 = 0.311
        T = 0.75 * 60 * 60  # Exponential decay time (0.75 hours in seconds)
        RE = constants.R_Earth  # Earth radius in cm
        E0 = constants.MC2  # Electron rest energy in MeV
        c = constants.c  # Speed of light in cm/s (SGS)
        e = constants.q  # Elementary charge in SGS

        # Calculate B
        B = dip.B(L)

        # Calculate Ems (electric field amplitude in mV/m)
        Ems = 0.26 * (kp - 1) + 0.1  # From Brautigam & Albert

        # Convert mu from MeV/G to erg/G
        mu_cgs = mu * 1.602e-6  # MeV to erg conversion factor

        # Calculate omega_D (relativistic drift frequency in rad/s)
        omega_D = ((3 * mu_cgs * c) / (e * L ** 2 * RE ** 2)) / ((1 + 2 * mu * B / E0) ** 0.5)


        # Calculate DllE (electrostatic radial diffusion coefficient in cm^2/s)
        dlle = (1 / 4) * (T * Ems ** 2 * 1e6 / B0 ** 2) / (1 + (omega_D * T / 2) ** 2) * L ** 6

        # Convert DllE from cm^2/s to 1/day
        dlle = dlle * (60 * 60 * 24) / RE ** 2  # Convert to 1/day

    if dll_type in {'M', 'm'}:
        return dllm
    elif dll_type in {'E', 'e'}:
        return dlle
    elif dll_type in {'ME', 'me', 'both', ''}:
        return dllm, dlle

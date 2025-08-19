import numpy as np
from rbamlib.models.ne import D2002


def D2006(L, MLT=0.0, r=None):
    r"""
    Denton et al. (2006) plasmaspheric electron density model with MLT dependence [#]_.

    Parameters
    ----------
    L : float or ndarray
        McIlwain :math:`L`-shell (dimensionless).
    MLT : float or ndarray, optional
        Magnetic local time in hours (0–24). Default is 0.
    r : float or ndarray, optional
        Geocentric distance along the field line in Earth radii. If provided,
        extend the equatorial value along the field line using :func:`D2002`.

    Returns
    -------
    ne : float or ndarray
        Electron density, cm⁻³. If ``r`` is provided, this is :math:`n_e(r)`;
        otherwise the equatorial value :math:`n_{e,eq}(L,MLT)`.

    Notes
    -----
    Denton et al. (2006) defines an *empirical equatorial plasmasphere model*
    with explicit MLT variation. Using :math:`\phi = 2\pi(\mathrm{MLT}-18)/24`, the two regimes are:

    **Eq. (2): inner plasmasphere** (:math:`L \le 3.2`)

    .. math::

       \log_{10} n_e = 5.25 - 0.82 L + 0.25 f(L) \cos\phi

    with

    .. math::

       f(L) =
       \begin{cases}
         \max(L-1.5, 0), & L \le 2.5, \\
         1, & L > 2.5
       \end{cases}

    **Eq. (3): outer plasmasphere** (:math:`L > 3.2`)

    .. math::

       \log_{10} n_e = 2.62 - 0.45 (L-3.2) + 0.12 (L-3.2)^2 + 0.25 \cos\phi

    The full Denton 2006 distribution along a field line is obtained by
    combining the equatorial density above with the Denton power law
    (see :func:`D2002`):

    .. math::

       n_e(r) = n_{e,eq}\left(\frac{R_{\max}}{r}\right)^{\alpha}, R_{\max} \approx L R_E

    Densities are in cm⁻³, distances in Earth radii.

    References
    ----------
    .. [#] Denton, R. E., Goldstein, J., Lee, D.-H., King, R. A., Dent, Z. C., Gallagher, D. L., et al. (2006). Realistic magnetospheric density model for 29 August 2000. Journal of Atmospheric and Solar-Terrestrial Physics, 68(6), 615–628. https://doi.org/10.1016/j.jastp.2005.11.009

    See Also
    --------
    D2002 : Denton field-aligned power law used when ``r`` is provided.
    """
    L = np.asarray(L, dtype=float)
    MLT = np.asarray(MLT, dtype=float)

    # broadcast
    Lb, MLTb = np.broadcast_arrays(L, MLT)

    phi = 2.0 * np.pi * (MLTb - 18.0) / 24.0
    log10_ne = np.empty_like(Lb, dtype=float)

    # inner plasmasphere: Eq. (2)
    inner = Lb <= 3.2
    if np.any(inner):
        Li = Lb[inner]
        fL = np.where(Li <= 2.5, np.clip(Li - 1.5, 0.0, None), 1.0)
        log10_ne[inner] = (5.25 - 0.82 * Li) + 0.25 * fL * np.cos(phi[inner])

    # outer plasmasphere: Eq. (3)
    outer = ~inner
    if np.any(outer):
        Lo = Lb[outer]
        dL = Lo - 3.2
        log10_ne[outer] = 2.62 - 0.45 * dL + 0.12 * (dL ** 2) + 0.25 * np.cos(phi[outer])

    ne_eq = 10.0 ** log10_ne

    if r is None:
        return ne_eq
    return D2002(r, ne_eq, L=L)
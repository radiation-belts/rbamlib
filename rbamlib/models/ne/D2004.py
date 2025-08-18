import numpy as np
from rbamlib.models.ne import D2002  # along-field power law (Denton 2002/2006)

def D2006(L, r=None, R13=13.0, doy=None, seasonal=False, fit="denton04", alpha=None):
    r"""
    Denton-style plasmaspheric electron density (equatorial or along field line) [#]_[#]_.

    By default this uses the **Denton et al. (2004)** equatorial fit (their Eq. 9):
    :math:`a_1 = 3.78`, :math:`a_2 = -0.324`, combined with the Carpenter–Anderson /
    Sheeley solar-cycle/seasonal bracket. You can set ``fit="ca1992"`` to use the original
    Carpenter–Anderson linear baseline instead (their :math:`a_1=3.9043`, :math:`a_2=-0.3145`).

    Parameters
    ----------
    L : float or ndarray
        McIlwain :math:`L`-shell (dimensionless). In Denton (2004), the linear term uses
        :math:`R_m = R_{\max}/R_E \approx L` (dipole approximation).
    r : float or ndarray, optional
        Geocentric distance along the field line in **Earth radii**. If provided, return
        :math:`n_e(r)` using :func:`rbamlib.models.ne.D2002`; otherwise return the
        equatorial value :math:`n_{e,eq}(L)`.
    R13 : float or ndarray, optional
        13-month smoothed sunspot number :math:`R`. Denton (2004) averaged their data at
        :math:`R\!\approx\!13` (default). To reproduce common legacy code, set ``R13=78``.
    doy : float or ndarray, optional
        Day of year (1–366). Only used when ``seasonal=True``.
    seasonal : bool, optional
        Include the Sheeley seasonal term if ``True``. Denton (2004) averaged over season
        and MLT, so the default is ``False``.
    fit : {"denton04","ca1992"}, optional
        Choice of linear-in-:math:`L` baseline inside :math:`\log_{10} n_{e,eq}`:
        - ``"denton04"`` (default): :math:`a_1=3.78`, :math:`a_2=-0.324` (Denton+ 2004 Eq. 9).
        - ``"ca1992"``: :math:`a_1=3.9043`, :math:`a_2=-0.3145` (Carpenter & Anderson 1992).
    alpha : float or ndarray, optional
        Field-aligned exponent passed to :func:`rbamlib.models.ne.D2002` when ``r`` is provided.
        If omitted, :func:`D2002` computes :math:`\alpha` from its standard parameterization.

    Returns
    -------
    ne : float or ndarray
        Electron density in **cm⁻³**. If ``r`` is given, this is :math:`n_e(r)`; otherwise
        the equatorial value :math:`n_{e,eq}(L)`. Shapes follow NumPy broadcasting.

    Notes
    -----
    **Equatorial law (Denton form):**

    Denton et al. (2004) fit the equatorial plasmaspheric density with the CA/Sheeley
    functional form, replacing :math:`L` by :math:`R_m=R_{\max}/R_E\approx L`:

    .. math::
        \log_{10} n_{e,eq}
        = \underbrace{a_1 + a_2\,R_m}_{\text{baseline}}
          \;+\; \Big[\,S(d) + 0.00127\,R - 0.0635\,\Big]\,
              \exp\!\left(-\frac{L-2}{1.5}\right), \qquad (5)

    with their best-fit coefficients (for :math:`2.5 \lesssim R_m \lesssim 8`):

    .. math::
        a_1 = 3.78 \pm 0.10, \qquad a_2 = -0.324 \pm 0.038. \qquad (9)

    The Carpenter–Anderson values (for comparison) are:

    .. math::
        a_1^{\mathrm{CA}} = 3.9043,\qquad a_2^{\mathrm{CA}} = -0.3145. \qquad (6)

    The seasonal term from Sheeley et al. (2001, Eq. 2) is

    .. math::
        S(d) = 0.15\!\left[
          \cos\!\frac{2\pi(d+9)}{365}
          - \tfrac{1}{2}\cos\!\frac{4\pi(d+9)}{365}
        \right].

    **Along-field variation (optional):**

    If a geocentric distance :math:`r` is provided, the returned density uses the Denton
    field-aligned power law (Denton 2002/2006):

    .. math::
        n_e(r) \;=\; n_{e,eq}\left(\frac{R_{\max}}{r}\right)^{\alpha}, \qquad R_{\max}\approx L\,R_E.

    **Units:** All densities are in **cm⁻³**; distances (``r``) in **Earth radii**.

    References
    ----------
    .. [#] Denton, R. E., Menietti, J. D., Goldstein, J., Young, S. L., & Anderson, R. R. (2004),
       *Electron density in the magnetosphere*, J. Geophys. Res., **109**, A09215,
       https://doi.org/10.1029/2003JA010245
    .. [#] Denton, R. E., et al. (2006), *Distribution of density along magnetospheric field lines*,
       J. Geophys. Res., **111**, A04213, https://doi.org/10.1029/2005JA011414
    .. [#] Carpenter, D. L., & Anderson, R. R. (1992), *An ISEE/whistler model of equatorial
       electron density in the magnetosphere*, J. Geophys. Res., **97**(A2), 1097–1108,
       https://doi.org/10.1029/91JA01548
    .. [#] Sheeley, B. W., et al. (2001), *An empirical plasmasphere and trough density model*,
       J. Geophys. Res., **106**(A11), 25631–25642, https://doi.org/10.1029/2000JA000286
    """
    L = np.asarray(L, dtype=float)
    R13 = np.asarray(R13, dtype=float)

    # --- coefficients a1, a2 (log10 baseline) ---
    fit_key = str(fit).lower()
    if fit_key in ("denton04", "d2004", "denton"):
        a1, a2 = 3.78, -0.324     # Denton et al. (2004), Eq. (9)
    elif fit_key in ("ca1992", "carpenteranderson", "ca"):
        a1, a2 = 3.9043, -0.3145  # Carpenter & Anderson (1992)
    else:
        raise ValueError("fit must be one of {'denton04','ca1992'}.")

    # --- optional seasonal term (Sheeley Eq. 2) ---
    S = 0.0
    if seasonal:
        if doy is None:
            raise ValueError("doy must be provided when seasonal=True.")
        d = np.asarray(doy, dtype=float)
        phi1 = 2.0 * np.pi * (d + 9.0) / 365.0
        S = 0.15 * (np.cos(phi1) - 0.5 * np.cos(2.0 * phi1))

    # --- CA/Sheeley solar-cycle/seasonal bracket with L-decay ---
    decay = np.exp(-(L - 2.0) / 1.5)
    bracket = (S + 0.00127 * R13 - 0.0635) * decay

    # --- equatorial log10(cm^-3) and density ---
    log10_ne_eq = (a1 + a2 * L) + bracket
    ne_eq = 10.0 ** log10_ne_eq  # cm^-3

    if r is None:
        return ne_eq

    # Along-field profile via Denton power law (D2002)
    return D2002(r, ne_eq, L=L, alpha=alpha)

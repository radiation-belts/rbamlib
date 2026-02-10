import numpy as np


def S2001(L, Lpp=None, kp=None, mask_invalid=True):
    r"""
    Equatorial electron density model of the plasmasphere and plasmatrough by Sheeley et al. :cite:yearpar:`sheeley:2001`.

    Parameters
    ----------
    L : float or ndarray
        McIlwain L-shell.
    Lpp : float or ndarray, optional
        Plasmapause location. If supplied, :math:`L \le L_{\mathrm{pp}}`
        uses the plasmasphere model and :math:`L > L_{\mathrm{pp}}` uses the trough
        model. If ``None``, all inputs default to the plasmasphere model.
    kp : float or ndarray, optional
        Kp index. Uses for plasmatrough calculation.
        If ``None``, the base model for plasmatrough density is returned
    mask_invalid : bool, optional
        If ``True`` (default), values outside :math:`3 \le L \le 7` are returned
        as ``NaN``. If ``False``, equations are evaluated without restriction.

    Returns
    -------
    ne : float or ndarray
        Equatorial electron density, :math:`\mathrm{cm}^{-3}`.

    Notes
    -----

    Model's valid range: :math:`3 \le L \le 7`.

    ----

    **Plasmasphere** (:math:`L \le L_{pp}`, Eq. 6):

    .. math::
        n_e(L) = 1390 (3/L)^{4.83}

    ----

    **Plasmatrough** (:math:`L > L_{pp}`, Eq. 7):

    Base model for plasmatrough density.

    .. math::

        n_e(L,\mathrm{LT}) = 124 \left(\frac{3}{L}\right)^{4.0}

    Plasmatrough density accounting for  Gallagher et al. (1998) :cite:p:`gallagher:1998` local time of maximum density.

    .. math::

       n_e(L,\mathrm{LT}) = 124 \left(\frac{3}{L}\right)^{4.0} + 36 \left(\frac{3}{L}\right)^{3.5}
         \cos\left(
           \frac{\pi}{12} \left[\mathrm{LT} - \left(7.7 \left(\tfrac{3}{L}\right)^{2.0} + 12\right)\right]
         \right)

    .. math::

        \mathrm{LT} = 0.145 K_p^2 - 2.63 K_p + 21.86

    ----
    """
    L = np.asarray(L, dtype=float)

    if Lpp is None:
        ne = S2001_plasmasphere(L)
    else:
        Lpp = np.asarray(Lpp, dtype=float)
        use_ps = L <= Lpp
        LT = _S2001_LT(kp)
        ne = np.where(use_ps, S2001_plasmasphere(L), S2001_trough(L, LT))

    if mask_invalid:
        valid = (L >= 3.0) & (L <= 7.0)
        ne = np.where(valid, ne, np.nan)

    return ne


# ----------------------------------------------------------------------
# Supporting helper functions
# ----------------------------------------------------------------------

def _S2001_LT(kp):
    r"""
    Local time of maximum trough density as a function of Kp (Gallagher et al., 1998) [#]_.

    Parameters
    ----------
    kp : float or ndarray
        Kp index.

    Returns
    -------
    LT : float or ndarray
        Local time of maximum density in hours (hours).

    Notes
    -----
    Empirical relation between Kp and local time (LT).

    .. math::
        \mathrm{LT} = 0.145 K_p^2 - 2.63 K_p + 21.86

    References
    ----------
    .. [#] Gallagher, D. L., et al., 1998,
    """
    if kp is None:
        return None

    kp = np.asarray(kp, dtype=float)
    LT = 0.145 * kp ** 2 - 2.63 * kp + 21.86

    # Preserve scalar return for scalar input
    if LT.shape == ():
        return float(LT)
    return LT


def S2001_plasmasphere(L):
    r"""
    Plasmasphere electron density model (Sheeley et al. 2001, Eq. 6).

    .. math::
        n_e(L) = 1390 (3/L)^{4.83}.
    """
    L = np.asarray(L, dtype=float)
    x = 3.0 / L
    return 1390.0 * x ** 4.83


def S2001_trough(L, LT=None):
    r"""
    Plasmatrough electron density model (Sheeley et al. 2001, Eq. 7).

    .. math::

       n_e(L,\mathrm{LT}) = 124 \left(\frac{3}{L}\right)^{4.0} + 36 \left(\frac{3}{L}\right)^{3.5}
         \cos\left(
           \frac{\pi}{12} \left[\mathrm{LT} - \left(7.7 \left(\tfrac{3}{L}\right)^{2.0} + 12\right)\right]
         \right)
    """
    L = np.asarray(L, dtype=float)
    x = 3.0 / L
    ne_base = 124.0 * x ** 4.0
    if LT is None:
        return ne_base

    LT = np.asarray(LT, dtype=float)
    phase = (LT - (7.7 * x ** 2.0 + 12.0)) * (np.pi / 12.0)
    return ne_base + 36.0 * x ** 3.5 * np.cos(phase)

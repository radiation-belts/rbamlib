import numpy as np


def CA1992(L, doy=None, R13=None, Lpp=None, MLT=None, Lppo=False):
    r"""
    Equatorial electron density :math:`n_e` (cm⁻³) by Carpenter & Anderson (1992) [#]_.

    Includes *plasmasphere* (L < Lpp), *plasmapause segment* (Lpp < L ≤ Lppo), and
    *plasma trough* (L ≥ Lpp or Lppo) regimes, with optional seasonal (`doy`) and solar–cycle
    corrections (`R13`). If ``Lpp`` is not supplied the plasmasphere value is returned.
    If `MLT` is provided, the resulting array is shaped as ``(len(L), len(MLT))``.
    `Lppo` is a flag which enables calculation of the plasmapause segment.

    - ``Lppo=False``: :math:`L \le L_{pp}` → plasmasphere; :math:`L > L_{pp}` → extended trough.
      (No plasmapause segment; this is a fast mask by :math:`L_{pp}`.)
    - ``Lppo=True``: solve an outer plasmapause :math:`L_{ppo}` by intersection and add the
      plasmapause segment for :math:`L_{pp} < L \le L_{ppo}`.


    Parameters
    ----------
    L : float or 1D ndarray
        McIlwain L shell.
    doy : int or ndarray, optional
        Day of year (1–366). If provided, seasonal terms are added in :math:`\log_{10} n_e`.
    R13 : float or ndarray, optional
        13-month smoothed sunspot number. If provided, solar–cycle term is added in :math:`\log_{10} n_e`.
    Lpp : float or 1D ndarray, optional
        Plasmapause inner limit. If array, it is a function of MLT (one per column).
    MLT : float or 1D ndarray, optional
        Magnetic local time in hours (0–24). If array, the output is 2-D of shape
        ``(len(L), len(MLT))``. If omitted while ``Lpp`` is given, MLT=0 is assumed.
    Lppo : bool, default=False
        If True, compute :math:`L_{ppo}` by intersection and include the plasmapause
        segment. If False, skip the segment and use a simple plasmasphere/trough mask.

    Returns
    -------
    ne : ndarray
        Equatorial electron density (cm⁻³). Shape:

        - ``(len(L),)`` if `MLT` is None or scalar,
        - ``(len(L), len(MLT))`` if `MLT` is a 1-D array.

    Notes
    -----

    ----

    **Plasmasphere** (:math:`L \le L_{pp}`):

    .. math::
        \log_{10} n_e = -0.3145L + 3.9043.

    Optional seasonal terms, using day of year :math:`\text{doy}=d`:

    .. math::
        \Delta_{\text{season}}(d) = 0.15 \cos\left( \frac{2\pi(d+9)}{365} \right) - 0.075 \cos\left( \frac{4\pi(d+9)}{365} \right).

    Optional solar–cycle term using 13-month smoothed sunspot number :math:`\text{R13}=\overline{R}`:

    .. math::

       \Delta_{\text{solar}}(\overline{R}) = (0.00127 \overline{R} - 0.0635) \exp(-(L-2)/1.5)

    The full expression with :math:`d`: and :math:`\overline{R}`: is

    .. math::

       \log_{10} n_e = -0.3145 L + 3.9043 + \Delta_{\text{season}}(d) + \Delta_{\text{solar}}(\overline{R})

    ----

    **Plasma trough** (:math:`L > L_{pp}`)

    .. math::

       n_e(L,MLT) = A(MLT) L^{-4.5} + (1 - \exp(-(L-2)/10)

       A(t) = \begin{cases}
           5800 + 300 MLT, & 0 \le MLT < 6, \\
           -800 + 1400 MLT, & 6 \le MLT \le 15
       \end{cases}

    If :math:`MLT > 15`, it is kept at constant :math:`MLT=15`.

    ----

    **Plasmapause segment** (:math:`L_{pp} < L \le L_{ppo}`, used only if ``Lppo=True``)

    Anchored to :math:`n_e(L_{pp})` with MLT-dependent decade slope.

    .. math::

       n_e(L,MLT) = n_e(L_{pp}) \times 10^{ - (L - L_{pp}) / s(MLT)}

       s(MLT) = \begin{cases}
           0.10, & 0 \le MLT < 6, \\
           0.10 + 0.01 (MLT-6), & 6 \le MLT \le 15
       \end{cases}


    The outer plasmapause :math:`L_{ppo}` is obtained by solving for the intersection
    between the plasmapause segment and the extended trough at the given MLT.

    If :math:`MLT > 15`, it is kept at constant :math:`MLT=15`.

    ----

    References
    ----------
    .. [#] Carpenter, D. L., & Anderson, R. R. (1992). An ISEE/Whistler model of equatorial electron density in the magnetosphere. *J. Geophys. Res.*, 97(A2), 1097–1108. https://doi.org/10.1029/91JA01548
    """

    # Normalize axes
    L = np.atleast_1d(np.asarray(L, dtype=float))
    nL = L.size

    # If no Lpp => pure plasmasphere (1D or tiled across MLT)
    if Lpp is None:
        ne1d = CA1992_plasmasphere(L, doy=doy, R13=R13)
        if MLT is None or np.ndim(MLT) == 0:
            return ne1d
        mlt1d = np.atleast_1d(np.asarray(MLT, dtype=float))
        return np.repeat(ne1d[:, None], mlt1d.size, axis=1)

    # Prepare MLT axis (mlt1d is the 1-D MLT array)
    mlt1d = np.array([0.0], dtype=float) if (MLT is None or np.ndim(MLT) == 0) \
        else np.atleast_1d(np.asarray(MLT, dtype=float))
    nmlt = mlt1d.size

    # Lpp as function of MLT
    Lpp_arr = np.atleast_1d(np.asarray(Lpp, dtype=float))
    if Lpp_arr.size == 1:
        Lpp_mlt = np.full(nmlt, float(Lpp_arr[0]), dtype=float)
    elif Lpp_arr.size == mlt1d.size:
        Lpp_mlt = Lpp_arr
    else:
        raise ValueError("Lpp must be scalar or have the same length as MLT.")

    # Allocate output (L, MLT)
    ne = np.empty((nL, nmlt), dtype=float)

    # Precompute plasmasphere along L for reuse
    ne_ps_L = CA1992_plasmasphere(L, doy=doy, R13=R13)

    # Column-wise evaluation over MLT
    for j in range(nmlt):
        Lpp_j = Lpp_mlt[j]
        mlt_val = mlt1d[j]

        if not Lppo:
            # Fast mask mode: plasmasphere for L <= Lpp, trough for L > Lpp
            ne[:, j] = np.where(L <= Lpp_j, ne_ps_L, CA1992_trough(L, mlt_val))
            continue

        # Full mode: include plasmapause segment and solve Lppo
        ne_Lpp = CA1992_plasmasphere(Lpp_j, doy=doy, R13=R13)
        Lppo_j = _CA1992_Lppo_solve(Lpp_j, mlt_val, doy=doy, R13=R13)

        ne_ppseg = CA1992_plasmapause(L, Lpp_j, ne_Lpp, mlt_val)
        ne_tr    = CA1992_trough(L, mlt_val)

        mask_ps = L <= Lpp_j
        mask_pp = (L > Lpp_j) & (L <= Lppo_j)
        mask_tr = L > Lppo_j

        col = np.empty(nL, dtype=float)
        col[mask_ps] = ne_ps_L[mask_ps]
        col[mask_pp] = ne_ppseg[mask_pp]
        col[mask_tr] = ne_tr[mask_tr]
        ne[:, j] = col

    return ne[:, 0] if nmlt == 1 else ne

# --------------------------------- Helpers ---------------------------------

def CA1992_plasmasphere(L, doy=None, R13=None):
    r"""
    Plasmasphere electron density (cm^-3), CA1992 base + optional corrections.

    Returns
    -------
    ne : ndarray
        Plasmaspheric electron density (cm^-3).

    Notes
    -----

    .. math::

       \log_{10} n_e = -0.3145 L + 3.9043
       + 0.15 \cos\left(\frac{2\pi(d+9)}{365}\right) - 0.075 \cos\left(\frac{4\pi(d+9)}{365}\right)
       + (0.00127 \overline{R} - 0.0635) e^{-(L-2)/1.5}
    """
    L = np.asarray(L, dtype=float)
    log10_ne = -0.3145 * L + 3.9043

    if doy is not None:
        d = np.asarray(doy, dtype=float)
        log10_ne = (log10_ne
                    + 0.15 * np.cos(2.0 * np.pi * (d + 9.0) / 365.0)
                    - 0.075 * np.cos(4.0 * np.pi * (d + 9.0) / 365.0))

    if R13 is not None:
        r = np.asarray(R13, dtype=float)
        log10_ne = log10_ne + (0.00127 * r - 0.0635) * np.exp(-(L - 2.0) / 1.5)

    ne = 10.0 ** log10_ne
    return ne

def CA1992_trough(L, MLT):
    r"""
    Extended plasma trough electron density (cm⁻3), Carpenter & Anderson (1992), Sec. 4.

    Notes
    -----

    .. math::
       n_e(L,MLT) = A(MLT) L^{-4.5} + (1 - \exp(-(L-2)/10)

       A(t) = \begin{cases}
           5800 + 300 MLT, & 0 \le MLT < 6, \\
           -800 + 1400 MLT, & 6 \le MLT \le 15
       \end{cases}

    - Inputs can be scalars or 1-D arrays. If `MLT` is an array, the output has
      shape ``(len(L), len(MLT))``.
    - `MLT` is *clipped* into ``[0, 15]`` per the model’s validity.

    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    t = np.atleast_1d(np.asarray(MLT, dtype=float))
    t = np.clip(t, 0.0, 15.0)

    A = np.where(t < 6.0, 5800.0 + 300.0 * t, -800.0 + 1400.0 * t)           # (T,)
    L_term = L ** (-4.5)                                                     # (N,)
    soft   = 1.0 - np.exp(-(L - 2.0) / 10.0)                                 # (N,)

    ne = L_term[:, None] * A[None, :] + soft[:, None]                        # (N,T)
    return ne[:, 0] if t.size == 1 else ne


def CA1992_plasmapause(L, Lpp, ne_Lpp, MLT):
    r"""
    Plasmapause segment (cm⁻3), anchored at :math:`L_{pp}`, with MLT-dependent decade slope.

    Notes
    -----

    .. math::

        n_e(L,MLT) = n_e(L_{pp}) \times 10^{ - (L - L_{pp}) / s(MLT)}

        s(MLT) = \begin{cases}
           0.10, & 0 \le MLT < 6, \\
           0.10 + 0.01 (MLT-6), & 6 \le MLT \le 15
        \end{cases}

    - Vectorized over ``(L, MLT)``; if `MLT` is an array, output shape is
      ``(len(L), len(MLT))``.
    - `MLT` is **clipped** into ``[0, 15]`` per the model’s validity.
    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    t = np.atleast_1d(np.asarray(MLT, dtype=float))
    t = np.clip(t, 0.0, 15.0)

    s = np.where(t < 6.0, 0.10, 0.10 + 0.01 * (t - 6.0))                     # (T,)
    ne = float(ne_Lpp) * 10.0 ** ( - (L[:, None] - float(Lpp)) / s[None, :] )
    return ne[:, 0] if t.size == 1 else ne


def _CA1992_Lppo_solve(Lpp, MLT, doy=None, R13=None, Lmax=8.0, ngrid=801):
    r"""
    Solve for outer plasmapause Lppo by intersecting the plasma trough with plasmapause segment.

    Returns
    -------
    Lppo : float
    """
    Lpp = float(Lpp)
    ne_Lpp = CA1992_plasmasphere(Lpp, doy=doy, R13=R13)
    Lg = np.linspace(Lpp, Lmax, int(ngrid))
    diff = CA1992_plasmapause(Lg, Lpp, ne_Lpp, MLT) - CA1992_trough(Lg, MLT)
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if idx.size > 0:
        i0 = idx[0]
        x0, x1 = Lg[i0], Lg[i0 + 1]
        y0, y1 = diff[i0], diff[i0 + 1]
        Lppo = x0 - y0 * (x1 - x0) / (y1 - y0)
        return float(Lppo)
    return float(Lg[np.argmin(np.abs(diff))])
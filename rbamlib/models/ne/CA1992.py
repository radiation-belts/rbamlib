import numpy as np


def CA1992(L, doy=None, R13=None, Lpp=None, MLT=None, Lppo=False):
    r"""
    Equatorial electron density :math:`n_e` (cm⁻³) by Carpenter & Anderson (1992) [#]_.

    Includes plasmasphere (L < Lpp), *plasmapause segment* (Lpp ≤ L ≤ Lppo), and
    plasma trough (L ≥ Lppo) regimes, with optional seasonal and solar–cycle
    corrections. If ``Lpp`` is not supplied the plasmasphere value is returned.
    If ``Lpp`` is supplied, the result becomes piecewise (by MLT column):

    - ``Lppo=False``: :math:`L \le L_{pp}` → plasmasphere; :math:`L > L_{pp}` → extended trough.
      (No plasmapause segment; this is a fast mask by :math:`L_{pp}`.)
    - ``Lppo=True``: solve an outer plasmapause :math:`L_{ppo}(t)` by intersection and add the
      **plasmapause segment** for :math:`L_{pp}(t) < L \le L_{ppo}(t)`.


    Parameters
    ----------
    L : float or 1D ndarray
        McIlwain L shell.
    doy : int or ndarray, optional
        Day of year (1–366). If provided, seasonal terms are added in :math:`\log_{10} n_e`.
    R13 : float or ndarray, optional
        13-month smoothed sunspot number. If provided, solar–cycle term with
        L-decay is added in :math:`\log_{10} n_e`.
    Lpp : float or 1D ndarray, optional
        Plasmapause inner limit. If array, it is a function of MLT (one per column).
    MLT : float or 1D ndarray, optional
        Magnetic local time in hours (0–24). If array, the output is 2-D of shape
        ``(len(L), len(mlt))``. If omitted while ``Lpp`` is given, MLT=0 is assumed.
    Lppo : bool, default=False
        If True, compute :math:`L_{ppo}(t)` by intersection and include the plasmapause
        segment. If False, skip the segment and use a simple plasmasphere/trough mask.

    Returns
    -------
    ne : ndarray
        Equatorial electron density (cm⁻³). Shape:
        - ``(len(L),)`` if `mlt` is None or scalar,
        - ``(len(L), len(mlt))`` if `mlt` is a 1-D array.

    Notes
    -----
    **Base CA92 relation (plasmasphere, equator):**

    .. math::
        \log_{10} n_e \;=\; -0.3145\,L + 3.9043.

    **Optional seasonal terms** (annual with December maximum and
    semiannual with equinoctial maxima), using day of year :math:`d`:

    .. math::
        \Delta_{\text{season}}(d) \;=\;
        0.15\,\cos\!\Bigl(\tfrac{2\pi(d+9)}{365}\Bigr)
        \;-\; 0.075\,\cos\!\Bigl(\tfrac{4\pi(d+9)}{365}\Bigr).

    **Optional solar–cycle term** using 13-month smoothed sunspot number
    :math:`\overline{R}`:

    .. math::
        \Delta_{\text{solar}}(\overline{R}) \;=\; 0.00127\,\overline{R} - 0.0635.

    The full expression used here is

    .. math::
        \log_{10} n_e \;=\; -0.3145\,L + 3.9043
        \;+\; \Delta_{\text{season}}(d)\;[\text{if } d\ \text{given}]
        \;+\; \Delta_{\text{solar}}(\overline{R})\;[\text{if } \overline{R}\ \text{given}].

    **Extended plasma trough**

    .. math::
        n_e(L,t) = A(t)\,L^{-4.5} + \big(1 - e^{-(L-2)/10}\big), \qquad
        A(t) = \begin{cases}
            5800 + 300\,t, & 0 \le t < 6, \\
            -800 + 1400\,t, & 6 \le t \le 15.
        \end{cases}

    **Plasmapause segment** (used only if ``Lppo=True``)

    .. math::
        n_e(L,t) = n_e(L_{pp}) \times 10^{(L - L_{pp}) / s(t)}, \qquad
        s(t) = \begin{cases}
            0.10, & 0 \le t < 6, \\
            0.10 + 0.01\,(t-6), & 6 \le t \le 15.
        \end{cases}

    **Saturated plasmasphere (equator):**

    .. math::
        \log_{10} n_e = -0.3145\,L + 3.9043
        + 0.15\cos\Big(\tfrac{2\pi(d+9)}{365}\Big)
        - 0.075\cos\Big(\tfrac{4\pi(d+9)}{365}\Big)
        + \big(0.00127\,\overline{R} - 0.0635\big) e^{-(L-2)/1.5}.

    **Plasmapause segment (Lpp ≤ L ≤ Lppo):** anchored to :math:`n_e(L_{pp})` and
    increases outward with an MLT‑dependent slope. The paper summarizes this with
    a decade (base‑10) exponential in :math:`L - L_{pp}`. We implement

    .. math::
        n_e(L) = n_e(L_{pp}) \times 10^{\,(L - L_{pp})/s(t)},

    with :math:`s(t) = 0.10` for :math:`0 \le t < 6` MLT, and
    :math:`s(t) = 0.10 + 0.01\,(t-6)` for :math:`6 \le t \le 15` MLT (linear-in‑t
    slope consistent with the summary panel). (Outside this range, we clamp to the
    nearest branch.)

    **Extended plasma trough (L ≥ Lppo):** eq. (5) style falloff with MLT dependence:

    .. math::
        n_e = (5800 + 300\,t)\,L^{-4.5} + \big(1 - e^{-(L-2)/10}\big), \quad 0 \le t < 6,\\
        n_e = (-800 + 1400\,t)\,L^{-4.5} + \big(1 - e^{-(L-2)/10}\big), \quad 6 \le t \le 15.

    The outer plasmapause :math:`L_{ppo}` is obtained by solving for the intersection
    between the plasmapause segment and the extended trough at the given MLT.

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

    For magnetic local time :math:`t` (hours), use the paper’s piecewise amplitude
    and add the soft offset:

    .. math::
        n_e(L,t) =
        \begin{cases}
            (5800 + 300\,t)\,L^{-4.5} + \bigl(1 - e^{-(L-2)/10}\bigr), & 0 \le t < 6,\\[4pt]
            (-800 + 1400\,t)\,L^{-4.5} + \bigl(1 - e^{-(L-2)/10}\bigr), & 6 \le t \le 15~.
        \end{cases}

    Notes
    -----
    - Inputs can be scalars or 1-D arrays. If `MLT` is an array, the output has
      shape ``(len(L), len(MLT))``.
    - `MLT` is **clipped** into ``[0, 15]`` per the model’s validity.

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

    .. math::
        s(t) = \begin{cases}
          0.10, & 0 \le t < 6,\\
          0.10 + 0.01\,(t-6), & 6 \le t \le 15,
        \end{cases}
        \qquad
        n_e(L,t) = n_e(L_{pp}) \times 10^{-\,(L - L_{pp}) / s(t)}.

    Notes
    -----
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
    Solve for outer plasmapause Lppo by intersecting the plasmapause segment
    with the extended trough at MLT.

    Returns
    -------
    Lppo : float
    """
    Lpp = float(Lpp)
    ne_Lpp = CA1992_plasmasphere(Lpp, doy=doy, R13=R13)
    Lg = np.linspace(Lpp, Lmax, int(ngrid))
    diff = CA1992_plasmapause(Lg, Lpp, ne_Lpp, MLT) - CA1992_trough(Lg, MLT)
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if idx.size:
        i0 = idx[0]
        x0, x1 = Lg[i0], Lg[i0 + 1]
        y0, y1 = diff[i0], diff[i0 + 1]
        Lppo = x0 - y0 * (x1 - x0) / (y1 - y0)
        return float(Lppo)
    return float(Lg[np.argmin(np.abs(diff))])
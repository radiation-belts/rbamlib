import numpy as np


def S1998(Phy, Bz, Pdyn):
    """
    Calculates the magnetopause location following Shue et al. (1998).

    Parameters
    ----------
    Phy : ndarray
        Angle between the Earth-Sun line in radians (1D array).
    Bz : ndarray
        IMF Bz in nT (1D array), must have the same size as Pdyn.
    Pdyn : ndarray
        Dynamic pressure in nPa (1D array), must have the same size as Bz.

    Returns
    -------
    r : ndarray
        Distance to the magnetopause from the Earth's center in Re.
        The result will be a 2D array of shape (len(Pdyn), len(Phy)).

    Notes
    -----
    The magnetopause location is calculated using the empirical model proposed by Shue et al. (1998):

    .. math::
        r_0(Bz) = \begin{cases}
        (11.4 + 0.013 \cdot Bz) \cdot Pdyn^{-1/6.6}, & \text{if } Bz \geq 0 \\
        (11.4 + 0.14 \cdot Bz) \cdot Pdyn^{-1/6.6}, & \text{if } Bz < 0
        \end{cases}

    .. math::
        r = r_0 \cdot \left( \frac{2}{1 + \cos(Phy)} \right)^{\alpha}

    where:

    .. math::
        \alpha = (0.58 - 0.010 \cdot Bz) \cdot (1 + 0.010 \cdot Pdyn)

    References
    ----------
    Shue, J.-H., Song, P., Russell, C. T., Steinberg, J. T., Chao, J. K., Zastenker, G., et al. (1998).
    Magnetopause location under extreme solar wind conditions.
    Journal of Geophysical Research, 103(A8), 17691–17700. https://doi.org/10.1029/98JA01103
    """

    # Ensure Bz and Pdyn are the same size
    if Bz.shape != Pdyn.shape:
        raise ValueError("Bz and Pdyn must have the same size")

    # Ensure arrays are floats
    Bz = Bz.astype(float)
    Pdyn = Pdyn.astype(float)
    Phy = Phy.astype(float)

    # Boolean mask for Bz >= 0
    Bz_poz = (Bz >= 0)

    # Initialize r0 with the same shape as Bz and Pdyn
    r0 = np.zeros_like(Bz)

    # Apply Shue model formulas for positive and negative Bz values
    r0[Bz_poz] = (11.4 + 0.013 * Bz[Bz_poz]) * (Pdyn[Bz_poz] ** (-1 / 6.6))
    r0[~Bz_poz] = (11.4 + 0.14 * Bz[~Bz_poz]) * (Pdyn[~Bz_poz] ** (-1 / 6.6))

    # Calculate r_alpha using the given formula
    r_alpha = (0.58 - 0.010 * Bz) * (1 + 0.010 * Pdyn)

    # Calculate the magnetopause distance for each combination of Pdyn and Phy
    r = np.outer(r0, (2 / (1 + np.cos(Phy))) ** r_alpha[:, np.newaxis])

    return r

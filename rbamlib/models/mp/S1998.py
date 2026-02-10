import numpy as np


def S1998(theta, Bz, Pdyn):
    r"""
    Calculates the magnetopause location following Shue et al. :cite:yearpar:`shue:1998`.

    Parameters
    ----------
    theta : ndarray
        Angle :math:`\theta` between the Earth-Sun line in radians (1D array).
    Bz : ndarray
        IMF :math:`B_z` in nT (1D array), must have the same size as :math:`P_{dyn}`.
    Pdyn : ndarray
        Dynamic pressure :math:`P_{dyn}` in nPa (1D array), must have the same size as :math:`B_z`.

    Returns
    -------
    r : ndarray
        Distance to the magnetopause from the Earth's center in :math:`R_E`.
        The result will be a 2D array of shape (len(Pdyn), len(theta)).

    Notes
    -----
    The magnetopause location is calculated using the empirical model proposed by Shue et al. (1998):

    .. math::
        r_0(B_z) = \begin{cases}
        (11.4 + 0.013 \cdot B_z) \cdot P_{dyn}^{-1/6.6}, & \text{if} B_z \geq 0 \\
        (11.4 + 0.14 \cdot B_z) \cdot P_{dyn}^{-1/6.6}, & \text{if} B_z < 0
        \end{cases}

    .. math::
        r = r_0 \cdot \left( \frac{2}{1 + \cos(\theta)} \right)^{\alpha}

    where:

    .. math::
        \alpha = (0.58 - 0.010 \cdot B_z) \cdot (1 + 0.010 \cdot P_{dyn})
    """

    # Ensure Bz and Pdyn are the same size
    if Bz.shape != Pdyn.shape:
        raise ValueError("Bz and Pdyn must have the same size")

    # Ensure arrays are floats
    Bz = Bz.astype(float)
    Pdyn = Pdyn.astype(float)
    theta = theta.astype(float)

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
    r = np.outer(r0, (2 / (1 + np.cos(theta))) ** r_alpha[:, np.newaxis])

    return r

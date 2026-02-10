
import numpy as np

def al_lc(L):
    r"""
    Equatorial loss-cone pitch angle, :math:`\alpha_{lc}`, in a dipole field
    at the surface of the planet, as a function of L-shell. 

    Parameters
    ----------
    L : float or ndarray
        McIlwain L-shell

    Returns
    -------
    float or ndarray
        Equatorial loss-cone pitch angle :math:`\alpha_{lc}` in radians.

    Notes
    -----
    **Deriviation** (based on chapter 3.4 of :cite:t:`roederer:2016`)

    1) Magnetic moment :math:`\mu` conservation:

    .. math::
        \mu = \frac{m v_\perp^2}{2B} = \text{const}
        \;\Rightarrow\;
        \sin^2\alpha_{lc} = \frac{B_{eq}}{B_{lc}}.


    :math:`m` - mass, :math:`v_\perp` - velocity, :math:`B` - magnetic field, :math:`\alpha` - pitch angle 

    2) Dipole magnitude:

    .. math::
        B(r,\lambda) = B_0\left(\frac{R}{r}\right)^3 \sqrt{1+3\sin^2\lambda}, \\
        B_{eq} = B_0 / L^3, (\lambda=0,\; r=L R)

    :math:`R` - planet radius, :math:`\lambda` - magnetic latitude
        
    3) Field line & mirror field:

    .. math::
        \begin{aligned}
        r &= L R \cos^2\lambda, \\ 
        r &= R \Rightarrow \cos^2\lambda_{lc} = \frac{1}{L_{lc}}, 
        \quad \sin^2\lambda_{lc} = 1-\frac{1}{L_{lc}}, \\
        B_{lc} &= B_0\sqrt{4-\frac{3}{L}}.
        \end{aligned}

    4) Loss-cone:

    .. math::
        \sin^2\alpha_{lc} = \frac{B_{eq}}{B_{lc}}
        = \frac{1}{L^3\sqrt{4-\frac{3}{L}}}, \qquad
        \alpha_{lc} = \arcsin\left[ \frac{1}{L^{3/2}(4-3/L)^{1/4}} \right].

    Examples
    --------
    >>> np.rad2deg(al_lc(4.0))
    5.34184...
    """
    L = np.asarray(L, dtype=np.float64)
  
    denom = L**3 * np.sqrt(4.0 - 3.0 / L)
  
    if np.any(denom <= 0):
        raise ValueError("Invalid L-shell: requires 4 - 3/L > 0.")
  
    val = 1.0 / np.sqrt(denom)  
    val = np.clip(val, 0.0, 1.0)  # numerical safety
  
    return np.arcsin(val)

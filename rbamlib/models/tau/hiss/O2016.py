
import numpy as np

def O2016(mlt, L, en, kp):
   r"""
   Calculates electron lifetime due to hiss waves following Orlova et al. :cite:yearpar:`orlova:2016` model.

   Parameters
   ----------
   mlt : float or ndarray
      Magnetic Local Time in hours.
   L : float or ndarray
      L-shell (McIlwain L), dimensionless. Valid range: :math:`1.5 \le L \le 5.5`.
      Inputs outside this range are clipped to the valid domain.
   en : float or ndarray
      Kinetic energy (MeV). Valid range: :math:`10^{-3} \le \mathrm{en} \le 10`.
      Inputs outside this range are clipped to the valid domain.
   kp : float or ndarray
      Planetary Kp index (dimensionless).

   Returns
   -------
   float or ndarray
      Electron lifetime (seconds).

   Notes
   -----   
   The parameterization is valid for:

   .. math::
        en = \log_{10}(en) \ge f(L), \quad
        f(L) = 0.1328 L^2 - 2.1463 L + 3.7857,
   
   where kinetic energy :math:`\mathrm{en} \in` [1 keV, 10 MeV] and :math:`L \in [1.5, 5.5]`.

   Electron lifetimes are parameterized as a function of L and kinetic energy :math:`E=log_{10}(\mathrm{en})`:
  
   .. math::
      \begin{aligned}
      \log_{10}\tau_{\rm av} = &
         a_1 + a_2 L + a_3 E + a_4 L^2 + a_5 L E + a_6 E^2 \\
         &+ a_7 L^3 + a_8 L^2 E + a_9 L E^2 + a_{10} E^3 \\
         &+ a_{11} L^4 + a_{12} L^3 E + a_{13} L^2 E^2 + a_{14} L E^3 \\
         &+ a_{15} E^4 + a_{16} L E^4 + a_{17} L^2 E^3 + a_{18} L^4 E \\
         &+ a_{19} L^5 + a_{20} E^5
      \end{aligned}

   The final lifetime is:
   
   .. math::
      \tau = \frac{\tau_{\rm av}}{g(\mathrm{MLT})\,h(\mathrm{kp})}.

   Extremely large :math:`\log_{10}\tau_{\rm av} > 10` is treated as ``inf``.
   """
   # ----- Input preparation & clipping (broadcast-friendly) -----
   L = np.clip(np.asarray(L, dtype=float), 1.5, 5.5)
   en = np.clip(np.asarray(en, dtype=float), 1e-3, 10.0)
   mlt = np.asarray(mlt, dtype=float)
   kp = np.asarray(kp, dtype=float)

   # ----- Helper computations -----
   # Compute log10(E) and validity mask
   log_en = np.log10(en)
   f_L = 0.1328 * L**2 - 2.1463 * L + 3.7857
   valid = log_en >= f_L
   
   # Compute MLT
   MLT = np.mod(mlt, 24.0)


   # Average lifetime surface tau_ave(L, log10(E))
   # Polynomial for tau_ave
   def _tau_ave(L, log10_en):
      # Coefficients from Orlova et al. (2016):
      coeffs = [
         77.323, -92.641, -55.754, 44.497, 48.981,
         8.9067, -10.704, -15.711, -3.3326, 1.5189,
         1.294, 2.2546, 0.31889, -0.85916, -0.22182,
         0.034318, 0.097248, -0.12192, -0.062765, 0.0063218
      ]
      (a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,
      a11,a12,a13,a14,a15,a16,a17,a18,a19,a20) = coeffs

      R2,R3,R4,R5 = L**2,L**3,L**4,L**5
      le,le2,le3,le4,le5 = log10_en,log10_en**2,log10_en**3,log10_en**4,log10_en**5

      log10_tau = (
         a1 + a2*L + a3*le + a4*R2 + a5*L*le + a6*le2 +
         a7*R3 + a8*R2*le + a9*L*le2 + a10*le3 +
         a11*R4 + a12*R3*le + a13*R2*le2 + a14*L*le3 +
         a15*le4 + a16*L*le4 + a17*R2*le3 + a18*R4*le +
         a19*R5 + a20*le5
      )
      
      # Avoid unrealistic huge exponentiations
      log10_tau = np.where(log10_tau > 10.0, np.inf, log10_tau)
      return np.power(10.0, log10_tau)

   tau_ave = _tau_ave(L, log_en)

   # MLT and Kp modifiers
   g = (10.0**(-0.007338*MLT**2 + 0.1773*MLT + 2.080)) / 782.3
   h = (10.0**(-0.01414*kp**2 + 0.2321*kp + 2.598)) / 1315.0

   # Apply validity mask
   tau = tau_ave / (g * h) * (24.0 * 3600.0)
   tau = np.where(valid, tau, np.nan)

   return tau
import numpy as np
from dataclasses import dataclass

@dataclass
class Coeff:
   """
   Data class for the polinomial coefficeints
   """
   a1: float = 0.0
   a2: float = 0.0
   a3: float = 0.0
   a4: float = 0.0
   a5: float = 0.0
   a6: float = 0.0
   a7: float = 0.0
   a8: float = 0.0
   a9: float = 0.0
   a10: float = 0.0

# ----------------------------
# Coefficients (Table S1)
# ----------------------------
# Each entry is a Coeff with keys 'a1'..'a10'.
# Numbers from the supporting materials (Table S1).  
COEFF_TABLE = {
    "day": {
        "1-10keV": Coeff(
            a1=-5.6829, a2=-0.0346, a3=-0.6096, a4=0.3570,
            a5=-0.1434, a6=0.0336, a7=1.1000, a8=-0.1170,
            a9=0.0095, a10=-0.0299
        ),
        "10-100keV": Coeff(
            a1=-0.1910, a2=0.6004, a3=0.0045, a4=1.1336,
            a5=0.1316, a6=-0.4351, a7=-0.0042, a8=0.7126,
            a9=-0.3492  # tenth coefficient not needed
        ),
        "0.1-0.4MeV": Coeff(
            a1=-0.6159, a2=1.5126, a3=1.1247, a4=-3.4438,
            a5=18.4731, a6=2.6884, a7=-7.9946, a8=-0.4590,
            a9=-6.5927, a10=0.5334
        ),
        "0.4-2.0MeV": Coeff(
            a1=-0.1697, a2=3.9861, a3=0.1233, a4=0.0264,
            a5=-2.4348, a6=13.9174 # Only six terms here
        ),
    },
    "night": {
        "1-10keV": Coeff(
            a1=-5.7729, a2=-0.0330, a3=-0.6771, a4=0.3936,
            a5=-0.1357, a6=0.0334, a7=1.0304, a8=-0.1193,
            a9=0.0095, a10=-0.0331
        ),
        "10-100keV": Coeff(
            a1=-0.4844, a2=0.9811, a3=-0.00014, a4=0.0103,
            a5=3.4253, a6=0.7607, a7=-4.6765, a8=13.3977,
            a9=-17.5595, a10=8.8706
        ),
        # Nightside above 0.1 MeV are not parameterized.
    }
}

def G2012(mlt, L, en, kp):
   r"""
   Calculates electron lifetime due to chorus waves following Gu et al. :cite:yearpar:`gu:2012` model, including correction :cite:p:`gu:2012:correction`.

   Parameters
   ----------
   mlt : float or ndarray
      Magnetic local time in hours.    
      Dayside is assumed for :math:`6 \le \mathrm{MLT} < 18`, nightside otherwise.  
   L : float or ndarray
      L-shell (McIlwain L), dimensionless. Valid: :math:`3 \le L \le 10`.
   en : float or ndarray
      Kinetic energy (MeV). Energy range: 1 keV–2 MeV      
   kp : float or ndarray
      Kp index (dimensionless). Scaling is applied for :math:`Kp \in [0,6]`.

   Returns
   -------
   float or ndarray
      Electron lifetime in seconds. Returns ``NaN`` if outside a parameterized bin (e.g.,
      nightside energies above 0.1 MeV).

   
   Notes
   -----
   **Parameterization.**
   Lifetimes are expressed as polynomials in :math:`L` and energy with coefficients
   :math:`a_1,\ldots,a_{10}` (Table S1), and equations from the Gu et al. (2012) 

   - **Dayside:** 1–10 keV (Eq. 1), 10–100 keV (Eq. 2), 0.1–0.4 MeV (Eq. 3), 0.4–2.0 MeV (Eq. 4).
   - **Nightside:** 1–10 keV (Eq. 5), 10–100 keV (Eq. 6, corrected).

   **Kp scaling and MLT coverage.**
   The final lifetime includes Kp‑dependent wave‑amplitude scaling and a factor of 4 to
   account for 25% MLT coverage (drift‑averaged presence of chorus):

   .. math::
      \tau(kp) = 4 \cdot \tau \cdot \left( \frac{B_w}{B_w(kp)} \right)^2 =
      \begin{cases}
         4\,\tau \left[2*10^\sqrt{0.73 + 0.91 kp}/{57.6}\right]^{-2}, & Kp \le 2+ \\
         4\,\tau \left[2*10^\sqrt{2.50 + 0.18 kp}/{57.6}\right]^{-2}, & 2+ < Kp \le 6
      \end{cases}
   """

   # ----------------------------
   # Inputs & MLT
   # ----------------------------
   # Convert inputs to numpy arrays and broadcast them to a common shape.
   L, en, kp, mlt = np.broadcast_arrays(
      np.asarray(L, dtype=float),
      np.asarray(en, dtype=float),   # MeV
      np.asarray(kp, dtype=float),
      np.asarray(mlt, dtype=float)
   )

   # Magnetic local time from longitude
   MLT =  mlt % 24.0
   is_day = (MLT >= 6.0) & (MLT < 18.0)

   # ----------------------------
   # Energy bin classification
   # ----------------------------
   # Convert MeV -> keV for keV bins
   en_keV = en * 1e3

   # Masks
   m_1_10keV = (en >= 0.001) & (en < 0.01)
   m_10_100keV = (en >= 0.01) & (en < 0.1)
   m_100_400keV = (en >= 0.1) & (en < 0.4)
   m_400_2000keV = (en >= 0.4) & (en <= 2.0)

   
   # Initialize t_base (days) with NaN
   t_base_days = np.full(en.shape, np.nan, dtype=float)

   # Apply per-hemisphere evaluators
   # Dayside
   mask_day = is_day
   sel = mask_day & m_1_10keV
   if np.any(sel):        
      t_base_days[sel] = tau_day_1_10keV(L[sel], en[sel])
   sel = mask_day & m_10_100keV
   if np.any(sel):        
      t_base_days[sel] = tau_day_10_100keV(L[sel], en[sel])
   sel = mask_day & m_100_400keV
   if np.any(sel):        
      t_base_days[sel] = tau_day_100_400keV(L[sel], en[sel])
   sel = mask_day & m_400_2000keV        
   if np.any(sel):        
      t_base_days[sel] = tau_day_400_2000keV(L[sel], en[sel])

   # Nightside (only up to 0.1 MeV)
   mask_night = ~is_day
   sel = mask_night & m_1_10keV
   if np.any(sel):        
      t_base_days[sel] = tau_night_1_10keV(L[sel], en[sel])
   sel = mask_night & m_10_100keV
   if np.any(sel):        
      t_base_days[sel] = tau_night_10_100keV(L[sel], en[sel])

   scale = kp_factor(kp)

   # Lifetimes in seconds
   t_sec = t_base_days * scale * 86400.0

   return t_sec

# ============================================================================
# Evaluate base lifetime in days
# ============================================================================
# Helper functions per energy bin & MLT.
def tau_day_1_10keV(L, en):
   # Eq. (1) for dayside 1–10 keV.   
   c = COEFF_TABLE["day"]["1-10keV"]
   # Convert MeV -> keV for keV bins!
   en = en * 1e3
   log10_t = (
         c.a1 * L**(-1) +
         c.a2 * L**2 +
         c.a3 * en**(-2) +
         c.a4 * en +
         c.a5 * L * en**(-1) +
         c.a6 * L**2 * en**(-1) +
         c.a7 * L**(-2) * en +
         c.a8 * L * en +
         c.a9 * L**2 * en +
         c.a10 * L**(-1) * en**2
   )
   return 10.0**log10_t

def tau_day_10_100keV(L, en):
   # Eq. (2) for dayside 10–100 keV.   
   c = COEFF_TABLE["day"]["10-100keV"]
   t = (
      c.a1 * L**(-1) +
      c.a2 * L**(-2) +
      c.a3 * L**(-2) * en**(-1) +
      c.a4 * L**(-1) * en +
      c.a5 * L * en +
      c.a6 * L**2 * en**2 +
      c.a7 * L**3 * en**2 +
      c.a8 * L**3 * en**3 +
      c.a9 * L**4 * en**4
   )
   return t

def tau_day_100_400keV(L, en):
   # Eq. (3) for dayside 100–400 keV.   
   c = COEFF_TABLE["day"]["0.1-0.4MeV"]
   t = (
      c.a1 * L**(-1) +
      c.a2 * L**(-2) +
      c.a3 * en +
      c.a4 * en**2 +
      c.a5 * en**3 +
      c.a6 * L**(-1) * en +
      c.a7 * L**(-2) * en +
      c.a8 * L * en**2 +
      c.a9 * L * en**4 +
      c.a10 * L**2 * en**4
   )
   return t

def tau_day_400_2000keV(L, en):
   # Eq. (4) for dayside 0.4–2 MeV.   
   c = COEFF_TABLE["day"]["0.4-2.0MeV"]
   t = (
      c.a1 * L**(-1) +
      c.a2 * L**(-2) +
      c.a3 * en +
      c.a4 * en**2 +
      c.a5 * L**(-1) * en +
      c.a6 * L**(-2) * en
   )
   return t

def tau_night_1_10keV(L, en):
   # Eq. (5) for nightside 1–10 keV.   
   c = COEFF_TABLE["night"]["1-10keV"]
   # Convert MeV -> keV for keV bins!
   en = en * 1e3
   log10_t = (
      c.a1 * L**(-1) +
      c.a2 * L**2 +
      c.a3 * en**(-2) +
      c.a4 * en +
      c.a5 * L * en**(-1) +
      c.a6 * L**2 * en**(-1) +
      c.a7 * L**(-2) * en +
      c.a8 * L * en +
      c.a9 * L**2 * en +
      c.a10 * L**(-1) * en**2
   )
   return 10.0**log10_t


def tau_night_10_100keV(L, en):
   # Eq. (6) corrected for nightside 10–100 keV.   
   c = COEFF_TABLE["night"]["10-100keV"]
   t = (
      c.a1 * L**(-1) +
      c.a2 * L**(-2) +
      c.a3 * L**2 +
      c.a4 * L**(-2) * en**(-1) +
      c.a5 * L**(-1) * en +
      c.a6 * L * en +
      c.a7 * L**2 * en**2 +
      c.a8 * L**3 * en**3 +
      c.a9 * L**4 * en**4 +
      c.a10 * L**5 * en**5
   )
   return t

# ============================================================================
# Kp scaling and main lifetime function
# ============================================================================
def kp_factor(Kp):
    """
    Compute the Kp-dependent scaling factor.
    (Eq. 15) with 25% MLT coverage factor of 4.  
    """
    Kp = np.asarray(Kp, dtype=float)
    expo_low = 0.73 + 0.91 * np.clip(Kp, 0, 2)
    expo_high = 2.50 + 0.18 * np.clip(Kp, 2, 6)
    
    BW_over_BWkp = np.where(
        Kp <= 2.0,
        np.sqrt(2*(10.0 ** expo_low) / 57.6),
        np.sqrt(2*(10.0 ** expo_high) / 57.6)
    )
    return 4.0 * (BW_over_BWkp ** (-2))
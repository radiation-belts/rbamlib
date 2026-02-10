"""
The `tau` provides lifetime models.

Models:
    - Lifetime within the loss cone (quarter of bounce period in dipole field) 

Additional models are grouped into packages that correspond to specific wave-particle interaction, like `chorus` or `hiss`.

Packages:
    - `chorus` : Lifetime models due to interaction with chorus waves
    - `hiss` : Lifetime models due to interaction with hiss waves
"""

from .tau_lc import tau_lc
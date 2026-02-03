"""
The `chorus` provides lifetime models due to interaction with chorus waves.

Models:
    - Gu et al., (2012) - Parameterized polynomial model
    - Wang et al., (2024) - Lookup table model
"""
from .G2012 import G2012
from .W2024 import W2024
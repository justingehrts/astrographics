"""Shared atmospheric extinction / air-mass helpers.

Used both for the sunset glow color (the Sun's light reddens and dims as
it passes through more atmosphere near the horizon) and, generalized here,
for individual stars at low altitude -- the same physics applies to any
point source, not just the Sun.
"""
import numpy as np

# Per-band extinction coefficients (roughly magnitudes per unit air mass
# at visible wavelengths for a clear, lightly hazy atmosphere).
BASE_EXTINCTION_R = 0.02
BASE_EXTINCTION_G = 0.04
BASE_EXTINCTION_B = 0.10


def air_mass(altitude_deg):
    """Kasten & Young (1989) air-mass approximation. Unlike the simple
    secant(zenith angle) formula, this stays finite and well-behaved all
    the way down to the horizon, which matters for a 0-40deg altitude tool."""
    alt = np.maximum(altitude_deg, 0.1)
    return 1.0 / (np.sin(np.radians(alt)) + 0.15 * (alt + 3.885) ** -1.253)


def rgb_transmission(altitude_deg, turbidity):
    """Per-channel atmospheric transmission factors (0-1) for light
    arriving from `altitude_deg`, scaled by the haze `turbidity` slider."""
    ext_factor = 1.0 + (turbidity - 1.0) * 0.25
    am = air_mass(altitude_deg)
    r = np.exp(-BASE_EXTINCTION_R * ext_factor * am)
    g = np.exp(-BASE_EXTINCTION_G * ext_factor * am)
    b = np.exp(-BASE_EXTINCTION_B * ext_factor * am)
    return r, g, b

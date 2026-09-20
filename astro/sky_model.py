"""Physically-based clear-daytime sky color, replacing the original
hand-tuned Gaussian Mie-glow + manually blended color-stop approximation.

Implements the Preetham, Shirley & Smits (2003) analytic daylight model
("A Practical Analytic Model for Daylight"): a Perez et al. luminance
distribution function whose coefficients are linear in atmospheric
turbidity, plus matching turbidity/sun-position-driven zenith luminance
and chromaticity. This covers the *daytime* sky (sun above the horizon);
twilight and night, which the model doesn't describe, keep the existing
hand-authored gradient/belt-of-Venus treatment in streamlit_app.py.
"""
import numpy as np

# Perez distribution coefficients, linear in turbidity T (Preetham et al.
# 2003, table 2).
_PEREZ_Y = dict(a=(0.1787, -1.4630), b=(-0.3554, 0.4275), c=(-0.0227, 5.3251),
                 d=(0.1206, -2.5771), e=(-0.0670, 0.3703))
_PEREZ_X = dict(a=(-0.0193, -0.2592), b=(-0.0665, 0.0008), c=(-0.0004, 0.2125),
                 d=(-0.0641, -0.8989), e=(-0.0033, 0.0452))
_PEREZ_y = dict(a=(-0.0167, -0.2608), b=(-0.0950, 0.0092), c=(-0.0079, 0.2102),
                 d=(-0.0441, -1.6537), e=(-0.0109, 0.0529))

# Zenith chromaticity matrices (Preetham et al. 2003, eq. 11).
_M_X = np.array([
    [0.00166, -0.00375, 0.00209, 0.0],
    [-0.02903, 0.06377, -0.03202, 0.00394],
    [0.11693, -0.21196, 0.06052, 0.25886],
])
_M_Y = np.array([
    [0.00275, -0.00610, 0.00317, 0.0],
    [-0.04214, 0.08970, -0.04153, 0.00516],
    [0.15346, -0.26756, 0.06670, 0.26688],
])

_XYZ_TO_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])


def _perez_coeffs(table, turbidity):
    return {k: a * turbidity + b for k, (a, b) in table.items()}


def _perez_f(theta_rad, gamma_rad, c):
    cos_theta = np.maximum(np.cos(theta_rad), 1e-3)
    return (1.0 + c["a"] * np.exp(c["b"] / cos_theta)) * (
        1.0 + c["c"] * np.exp(c["d"] * gamma_rad) + c["e"] * np.cos(gamma_rad) ** 2
    )


def _zenith_luminance(turbidity, sun_zenith_rad):
    chi = (4.0 / 9.0 - turbidity / 120.0) * (np.pi - 2.0 * sun_zenith_rad)
    return (4.0453 * turbidity - 4.9710) * np.tan(chi) - 0.2155 * turbidity + 2.4192


def _zenith_chromaticity(matrix, turbidity, sun_zenith_rad):
    t_vec = np.array([turbidity ** 2, turbidity, 1.0])
    theta_vec = np.array([sun_zenith_rad ** 3, sun_zenith_rad ** 2, sun_zenith_rad, 1.0])
    return t_vec @ matrix @ theta_vec


def daytime_sky_rgb(theta_rad, gamma_rad, sun_altitude_deg, turbidity):
    """Clear-sky color at points `theta_rad` (zenith angle from straight
    up) and `gamma_rad` (angular distance from the sun), for the sun at
    `sun_altitude_deg`. Returns linear RGB with roughly unit magnitude
    near the zenith -- callers should clip to [0, 1] before display, since
    the circumsolar glow legitimately exceeds 1 close to the sun."""
    sun_zenith_rad = np.radians(np.clip(90.0 - sun_altitude_deg, 1.0, 89.0))

    coeffs_Y = _perez_coeffs(_PEREZ_Y, turbidity)
    coeffs_x = _perez_coeffs(_PEREZ_X, turbidity)
    coeffs_y = _perez_coeffs(_PEREZ_y, turbidity)

    f_Y = _perez_f(theta_rad, gamma_rad, coeffs_Y) / _perez_f(0.0, sun_zenith_rad, coeffs_Y)
    f_x = _perez_f(theta_rad, gamma_rad, coeffs_x) / _perez_f(0.0, sun_zenith_rad, coeffs_x)
    f_y = _perez_f(theta_rad, gamma_rad, coeffs_y) / _perez_f(0.0, sun_zenith_rad, coeffs_y)

    xz = _zenith_chromaticity(_M_X, turbidity, sun_zenith_rad)
    yz = _zenith_chromaticity(_M_Y, turbidity, sun_zenith_rad)

    # Relative luminance (1.0 at the zenith itself). Preetham's absolute Yz
    # is a photometric quantity this app has no use for -- only the
    # dimensionless *shape* of the luminance gradient (f_Y) matters here.
    Y = np.clip(f_Y, 0.0, None)
    x = xz * f_x
    y = np.maximum(yz * f_y, 1e-3)

    X = (x / y) * Y
    Z = ((1.0 - x - y) / y) * Y
    xyz = np.stack([X, Y, Z], axis=-1)
    rgb = np.maximum(xyz @ _XYZ_TO_SRGB.T, 0.0)

    # Reinhard tone-mapping (by each pixel's max channel, to preserve hue)
    # instead of a hard clip. The Perez luminance gradient is unbounded and,
    # near the horizon especially, is well known to overshoot real sky
    # brightness (the classic Preetham "horizon overshoot" limitation) --
    # a hard clip would flatten both the sun-glow and the horizon band to
    # the same blown-out white instead of a distinguishable gradient.
    channel_max = np.maximum(rgb.max(axis=-1, keepdims=True), 1e-6)
    rgb *= (channel_max / (1.0 + channel_max)) / channel_max

    return np.clip(rgb, 0.0, 1.0)

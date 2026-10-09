"""Point-source appearance shared by stars and planets: color from the
Hipparcos B-V index, marker size from apparent magnitude, and how much of
the sky is visible through twilight.
"""
import numpy as np
import pandas as pd

# Field positions in the fixed-format Hipparcos main catalog (ESA 1997,
# hip_main.dat, '|'-separated): H1 = HIP number, H37 = B-V color index.
_HIP_FIELD = 1
_BV_FIELD = 37


def load_bv_series(fobj):
    """B-V color index per HIP number, read from the same hip_main.dat file
    Skyfield's hipparcos.load_dataframe() parses (which skips this column).
    Blank B-V fields come back as NaN."""
    fobj.seek(0)
    magic = fobj.read(2)
    fobj.seek(0)
    df = pd.read_csv(
        fobj, sep="|", header=None, usecols=[_HIP_FIELD, _BV_FIELD],
        compression="gzip" if magic == b"\x1f\x8b" else None,
    )
    hip = pd.to_numeric(df[_HIP_FIELD], errors="coerce")
    bv = pd.to_numeric(df[_BV_FIELD], errors="coerce")
    has_hip = hip.notna()
    return pd.Series(bv[has_hip].values, index=hip[has_hip].astype("int64").values, name="bv")


def bv_to_rgb(bv, saturation=0.8):
    """Approximate perceived star color (RGB in 0-1, brightest channel = 1)
    for B-V index: B-V -> effective temperature (Ballesteros 2012), then
    temperature -> blackbody sRGB (Helland's fit), blended toward white by
    `1 - saturation` since naked-eye star colors are subtle. NaN -> white."""
    bv = np.asarray(bv, dtype=float)
    missing = np.isnan(bv)
    bv = np.clip(np.where(missing, 0.65, bv), -0.4, 2.0)
    temp_k = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
    t = np.clip(temp_k, 1000.0, 40000.0) / 100.0

    r = np.where(t <= 66, 255.0, 329.698727446 * np.power(np.maximum(t - 60, 1e-6), -0.1332047592))
    g = np.where(
        t <= 66,
        99.4708025861 * np.log(t) - 161.1195681661,
        288.1221695283 * np.power(np.maximum(t - 60, 1e-6), -0.0755148492),
    )
    b = np.where(
        t >= 66, 255.0,
        np.where(t <= 19, 0.0, 138.5177312231 * np.log(np.maximum(t - 10, 1e-6)) - 305.0447927307),
    )
    rgb = np.clip(np.stack([r, g, b], axis=-1), 0.0, 255.0) / 255.0
    rgb = rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1e-6)
    rgb = 1.0 - saturation * (1.0 - rgb)
    rgb[missing] = 1.0
    return rgb


def marker_size(mag):
    """Scatter marker area (points^2) for apparent magnitude. Area grows by
    10^0.2 (~1.58x) per magnitude -- gentler than true flux (2.512x), so a
    -4 Venus doesn't swamp the frame, but with far more range than a linear
    size scale (Sirius ends up ~4x the diameter of a mag 4.5 star)."""
    mag = np.asarray(mag, dtype=float)
    return np.clip(1.2 * np.power(10.0, 0.2 * (6.0 - mag)), 1.0, 70.0)


# Naked-eye zenith limiting magnitude vs. Sun altitude, through twilight
# into full darkness. Rough empirical values: Venus is visible around
# sunset, Jupiter/Sirius by sun -2 to -3, 1st-magnitude stars by the end of
# civil twilight (-6), ~mag 5 by the end of nautical twilight (-12).
_TWILIGHT_SUN_ALT = [-18.0, -15.0, -12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 5.0]
_TWILIGHT_LIMIT = [6.5, 5.8, 4.8, 4.0, 3.0, 2.0, 0.5, -0.5, -1.5, -2.5, -3.5, -4.0]

FADE_WIDTH_MAG = 0.6


def twilight_limiting_magnitude(sun_alt_deg):
    return float(np.interp(sun_alt_deg, _TWILIGHT_SUN_ALT, _TWILIGHT_LIMIT))


def visibility_alpha(effective_mag, limit_mag):
    """Opacity for a point source: 0 when fainter than the limit, fading in
    over FADE_WIDTH_MAG (no hard pop-in as twilight deepens), and dimmer for
    faint sources than bright ones."""
    effective_mag = np.asarray(effective_mag, dtype=float)
    headroom = limit_mag - effective_mag
    fade_in = np.clip(headroom / FADE_WIDTH_MAG, 0.0, 1.0)
    brightness = 0.5 + 0.5 * np.clip(headroom / 3.0, 0.0, 1.0)
    return fade_in * brightness


# Naked-eye planet colors (before atmospheric reddening near the horizon).
PLANET_COLORS = {
    "Mercury": (0.93, 0.90, 0.85),
    "Venus": (1.00, 0.99, 0.94),
    "Mars": (1.00, 0.62, 0.40),
    "Jupiter": (1.00, 0.96, 0.86),
    "Saturn": (1.00, 0.91, 0.70),
}

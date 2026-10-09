"""Physically-based sky brightness and color, from full daylight through
twilight into night, replacing the Preetham daytime model plus the
hand-authored twilight/night gradients it was stitched to.

Single scattering by air molecules (Rayleigh), haze (Mie) and ozone
absorption, ray-marched around a spherical Earth for each pixel of the
sky grid. Because the Earth's shadow is part of the geometry, sunset
color, the twilight arch, Earth's shadow and the Belt of Venus all fall
out of the same calculation instead of being painted on. Output is in
real luminance units (cd/m^2), since the light source is the Sun's (or
Moon's) actual illuminance at the top of the atmosphere.
"""
from functools import lru_cache

import numpy as np

EARTH_RADIUS_M = 6360e3
ATMOSPHERE_TOP_M = 100e3
OBSERVER_HEIGHT_M = 2.0

# Channels are evaluated at effective wavelengths for sRGB's primaries
# (red ~610, green ~550, blue ~465 nm), not the spectral extremes: red at
# 610 nm sits right on ozone's Chappuis absorption peak, which is what
# turns the real twilight zenith blue rather than purple.
# Molecular scattering at sea level (lambda^-4 from 13.558e-6 at 550 nm).
RAYLEIGH_BETA = np.array([8.96e-6, 13.558e-6, 26.5e-6])
RAYLEIGH_SCALE_HEIGHT_M = 8000.0

# Aerosol (haze) scattering: optical depth scales with the turbidity
# slider (Preetham-style T: tau_aerosol ~ (T - 1) x tau_rayleigh(550nm)),
# mild Angstrom wavelength dependence, ~10% absorbing.
MIE_SCALE_HEIGHT_M = 1200.0
MIE_ANGSTROM = np.array([0.87, 1.0, 1.24])
MIE_SINGLE_SCATTER_ALBEDO = 0.9
MIE_G = 0.76

# Ozone (Chappuis band) absorption at the layer's peak density, scaled
# from its 550 nm value by the band's cross-section at each wavelength.
OZONE_ABSORPTION = np.array([2.85e-6, 1.881e-6, 0.28e-6])

REC709_LUMA = np.array([0.2126, 0.7152, 0.0722])
SOLAR_ILLUMINANCE_LUX = 1.27e5


def _ozone_density(h):
    return np.maximum(0.0, 1.0 - np.abs(h - 25e3) / 15e3)


def _densities(h):
    return np.exp(-h / RAYLEIGH_SCALE_HEIGHT_M), np.exp(-h / MIE_SCALE_HEIGHT_M), _ozone_density(h)


def _sphere_exit_distance(r, mu, radius):
    """Distance along a ray starting at radius r with direction cosine mu
    (relative to local vertical) to where it leaves a sphere of `radius`."""
    disc = r * r * (mu * mu - 1.0) + radius * radius
    return -r * mu + np.sqrt(np.maximum(disc, 0.0))


def _ray_hits_ground(r, mu):
    return (mu < 0.0) & (r * r * (mu * mu - 1.0) + EARTH_RADIUS_M ** 2 >= 0.0)


_LUT_HEIGHTS = 64
_LUT_MUS = 512
_LUT_MU_MIN = -0.20
_SHADOWED_COLUMN_M = 1e9


@lru_cache(maxsize=1)
def _light_column_lut():
    """Column amounts (meters of sea-level-equivalent air) of Rayleigh,
    Mie and ozone material between a point at height h and the top of the
    atmosphere toward a light at direction cosine mu. Rays that hit the
    ground are in the Earth's shadow and get an effectively infinite
    column. Indexed [height, mu, species]. Computed once."""
    u = np.linspace(0.0, 1.0, _LUT_HEIGHTS)
    heights = ATMOSPHERE_TOP_M * u * u  # denser near the ground
    mus = np.linspace(_LUT_MU_MIN, 1.0, _LUT_MUS)
    H, MU = np.meshgrid(heights, mus, indexing="ij")
    r = EARTH_RADIUS_M + H
    length = _sphere_exit_distance(r, MU, EARTH_RADIUS_M + ATMOSPHERE_TOP_M)

    n = 96
    s = (np.arange(n) + 0.5) / n
    t = length[..., None] * s * s
    dt = length[..., None] * (2.0 * s / n)
    sin2 = 1.0 - MU[..., None] ** 2
    rr = np.sqrt((r[..., None] + t * MU[..., None]) ** 2 + (t ** 2) * sin2)
    hh = rr - EARTH_RADIUS_M
    rho_r, rho_m, rho_o = _densities(hh)
    cols = np.stack([(rho_r * dt).sum(-1), (rho_m * dt).sum(-1), (rho_o * dt).sum(-1)], axis=-1)

    cols[_ray_hits_ground(r, MU)] = _SHADOWED_COLUMN_M
    return heights, mus, cols


def _lookup_light_columns(h, mu):
    heights, mus, cols = _light_column_lut()
    hi = np.sqrt(np.clip(h, 0.0, ATMOSPHERE_TOP_M) / ATMOSPHERE_TOP_M) * (_LUT_HEIGHTS - 1)
    mi = (np.clip(mu, _LUT_MU_MIN, 1.0) - _LUT_MU_MIN) / (1.0 - _LUT_MU_MIN) * (_LUT_MUS - 1)
    h0 = np.clip(np.floor(hi).astype(int), 0, _LUT_HEIGHTS - 2)
    m0 = np.clip(np.floor(mi).astype(int), 0, _LUT_MUS - 2)
    fh = (hi - h0)[..., None]
    fm = (mi - m0)[..., None]
    c00, c01 = cols[h0, m0], cols[h0, m0 + 1]
    c10, c11 = cols[h0 + 1, m0], cols[h0 + 1, m0 + 1]
    out = (c00 * (1 - fh) * (1 - fm) + c01 * (1 - fh) * fm + c10 * fh * (1 - fm) + c11 * fh * fm)
    below = mu < _LUT_MU_MIN
    out[below] = _SHADOWED_COLUMN_M
    return out


def mie_beta(turbidity):
    """Aerosol scattering coefficient at sea level per color channel."""
    aerosol_optical_depth_550 = 0.02 + max(turbidity - 1.0, 0.0) * 0.06
    return aerosol_optical_depth_550 / MIE_SCALE_HEIGHT_M * MIE_ANGSTROM


def unit_vectors(alt_deg, az_deg):
    """East-north-up unit vectors for altitude/azimuth (degrees)."""
    alt, az = np.radians(alt_deg), np.radians(az_deg)
    return np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], axis=-1)


def scattered_luminance(view_dirs, source_dir, source_illuminance_lux, turbidity, n_view=40):
    """Single-scattered sky radiance (linear RGB, cd/m^2) seen along
    `view_dirs` (..., 3) for a light source in direction `source_dir` (3,)
    with the given illuminance at the top of the atmosphere."""
    beta_m = mie_beta(turbidity)
    beta_m_ext = beta_m / MIE_SINGLE_SCATTER_ALBEDO
    r0 = EARTH_RADIUS_M + OBSERVER_HEIGHT_M
    mu_view = view_dirs[..., 2]
    length = _sphere_exit_distance(r0, mu_view, EARTH_RADIUS_M + ATMOSPHERE_TOP_M)
    length = np.where(_ray_hits_ground(r0, mu_view), 0.0, length)

    cos_theta = view_dirs @ source_dir
    phase_r = 3.0 / (16.0 * np.pi) * (1.0 + cos_theta ** 2)
    g = MIE_G
    phase_m = (3.0 / (8.0 * np.pi) * (1 - g * g) * (1 + cos_theta ** 2)
               / ((2 + g * g) * (1 + g * g - 2 * g * cos_theta) ** 1.5))

    shape = view_dirs.shape[:-1]
    sum_r = np.zeros(shape + (3,))
    sum_m = np.zeros(shape + (3,))
    tau_view = np.zeros(shape + (3,))
    origin = np.array([0.0, 0.0, r0])

    s_prev = 0.0
    for i in range(n_view):
        s = ((i + 1) / n_view) ** 2  # samples denser near the observer
        s_mid = 0.5 * (s + s_prev)
        dt = (length * (s - s_prev))[..., None]
        s_prev = s
        p = origin + view_dirs * (length * s_mid)[..., None]
        r = np.linalg.norm(p, axis=-1)
        h = r - EARTH_RADIUS_M
        rho_r, rho_m, rho_o = _densities(h)
        ext = RAYLEIGH_BETA * rho_r[..., None] + beta_m_ext * rho_m[..., None] + OZONE_ABSORPTION * rho_o[..., None]
        tau_here = tau_view + 0.5 * ext * dt
        tau_view = tau_view + ext * dt

        mu_light = (p @ source_dir) / r
        cols = _lookup_light_columns(h, mu_light)
        tau_light = (RAYLEIGH_BETA * cols[..., 0:1] + beta_m_ext * cols[..., 1:2]
                     + OZONE_ABSORPTION * cols[..., 2:3])
        atten = np.exp(-(tau_here + tau_light))
        sum_r += atten * rho_r[..., None] * dt
        sum_m += atten * rho_m[..., None] * dt

    radiance = (sum_r * RAYLEIGH_BETA * phase_r[..., None] + sum_m * beta_m * phase_m[..., None])
    return radiance * source_illuminance_lux


def luminance(rgb):
    return rgb @ REC709_LUMA


def surface_brightness_mag(luminance_cd_m2):
    """cd/m^2 -> V-band-ish surface brightness in mag/arcsec^2."""
    return -2.5 * np.log10(np.maximum(luminance_cd_m2, 1e-12) / 10.8e4)


# --- Multiple scattering ----------------------------------------------------
# Single scattering alone underestimates the daytime sky and, below about
# sun -6deg, collapses far faster than the real sky does: in deep twilight
# the sky is lit mostly by light scattered more than once (from the bright
# arch toward the sun), filtered blue by ozone. Daytime: add a fraction of
# the single-scattered field. Twilight: add a blue, gently directional
# term whose zenith level makes the total follow measured twilight zenith
# brightness (from ~13.3 mag/arcsec^2 at the end of civil twilight to the
# night floor by the end of astronomical twilight).
DAYTIME_MULTIPLE_SCATTER_FRACTION = 0.5
TWILIGHT_DIFFUSE_FRACTION = 0.35
_TWILIGHT_SUN_ALT = np.array([-18.0, -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0])
_TWILIGHT_ZENITH_MAG = np.array([30.0, 22.1, 20.6, 19.2, 17.5, 15.6, 13.3, 10.3])


def _twilight_zenith_target(sun_alt_deg):
    """Measured-twilight zenith luminance (cd/m^2), sun component only."""
    if sun_alt_deg > _TWILIGHT_SUN_ALT[-1]:
        return 0.0
    mag = np.interp(sun_alt_deg, _TWILIGHT_SUN_ALT, _TWILIGHT_ZENITH_MAG)
    return float(10.8e4 * 10 ** (-0.4 * mag))


def _twilight_ms_color():
    c = RAYLEIGH_BETA / RAYLEIGH_BETA.max() * np.exp(-OZONE_ABSORPTION * 3e5)
    return c / (c @ REC709_LUMA)


def _twilight_ms_shape(alt_deg, az_deg, sun_az_deg):
    """Relative brightness of the multiply-scattered twilight glow: higher
    toward the sun's azimuth and toward the horizon; 1.0 at the zenith."""
    cos_alt = np.cos(np.radians(alt_deg))
    toward_sun = 0.5 * (1.0 + np.cos(np.radians(az_deg - sun_az_deg)))
    azimuthal = 0.6 + 0.4 * (cos_alt * toward_sun + (1.0 - cos_alt) * 0.5)
    vertical = 1.0 + 0.4 * np.exp(-np.maximum(alt_deg, 0.0) / 12.0)
    return azimuthal * vertical / 0.8


def _daytime_weight(source_alt_deg):
    return float(np.clip((source_alt_deg + 2.0) / 6.0, 0.0, 1.0))


def _source_sky(alt_deg, az_deg, source_alt_deg, source_az_deg, illuminance_lux, turbidity,
                twilight_tail):
    """Single + multiple scattering from one light source, plus the same at
    the zenith (for exposure and star-visibility references)."""
    dirs = unit_vectors(alt_deg, az_deg)
    src = unit_vectors(np.array(source_alt_deg, float), np.array(source_az_deg, float))
    zen = unit_vectors(np.array([90.0]), np.array([0.0]))
    grid = scattered_luminance(dirs, src, illuminance_lux, turbidity)
    zenith = scattered_luminance(zen, src, illuminance_lux, turbidity)[0]

    day_ms = 1.0 + DAYTIME_MULTIPLE_SCATTER_FRACTION * _daytime_weight(source_alt_deg)
    grid, zenith = grid * day_ms, zenith * day_ms

    # Diffuse blue skylight from late afternoon on: it's what keeps Earth's
    # shadow blue-grey rather than black, and mixes with the reddened belt
    # above it to make the Belt of Venus pink.
    color = _twilight_ms_color()
    shape = _twilight_ms_shape(alt_deg, az_deg, source_az_deg)[..., None]
    diffuse = TWILIGHT_DIFFUSE_FRACTION * float(np.clip((6.0 - source_alt_deg) / 8.0, 0.0, 1.0)) * luminance(zenith)
    grid, zenith = grid + diffuse * color * shape, zenith + diffuse * color

    if twilight_tail:
        scale = illuminance_lux / SOLAR_ILLUMINANCE_LUX
        missing = max(_twilight_zenith_target(source_alt_deg) * scale - luminance(zenith), 0.0)
        if missing > 0.0:
            grid = grid + missing * color * shape
            zenith = zenith + missing * color
    return grid, zenith


# --- Night floor --------------------------------------------------------------
NATURAL_NIGHT_ZENITH_MAG = 21.8
_AIRGLOW_HEIGHT_M = 300e3
_NIGHT_COLOR = np.array([0.33, 0.50, 1.0])
# Skyglow looks warm (sodium/LED-orange) low over a city but close to a
# cool grey overhead, where the eye's night vision dominates.
_LIGHT_POLLUTION_HORIZON_COLOR = np.array([1.0, 0.80, 0.58])
_LIGHT_POLLUTION_ZENITH_COLOR = _NIGHT_COLOR

# Star Visibility Limit slider -> realistic moonless zenith sky brightness
# (mag/arcsec^2) for that kind of site, from inner city to pristine. The
# slider itself still caps which stars are drawn (it doubles as a
# declutter control); this only sets how bright the night sky glows.
_SLIDER_STEPS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
_SLIDER_SITE_MAG = [17.0, 17.6, 18.3, 19.0, 19.8, 20.6, 21.3, 21.75]


def _mag_to_cd(mag):
    return 10.8e4 * 10 ** (-0.4 * mag)


def _unit_luma(color):
    return color / (color @ REC709_LUMA)


def night_floor(alt_deg, star_visibility, turbidity):
    """Natural airglow (brighter toward the horizon, van Rhijn effect) plus
    light-pollution skyglow (much brighter toward the horizon), in linear
    RGB cd/m^2. Returns (natural, light_pollution) grids."""
    alt = np.radians(np.maximum(alt_deg, 0.0))
    ratio = EARTH_RADIUS_M / (EARTH_RADIUS_M + _AIRGLOW_HEIGHT_M)
    van_rhijn = np.minimum(1.0 / np.sqrt(1.0 - (ratio * np.cos(alt)) ** 2), 2.5)
    natural_cd = _mag_to_cd(NATURAL_NIGHT_ZENITH_MAG)
    natural = (natural_cd * van_rhijn)[..., None] * _unit_luma(_NIGHT_COLOR)

    site_mag = float(np.interp(star_visibility, _SLIDER_STEPS, _SLIDER_SITE_MAG))
    lp_zenith_cd = max(_mag_to_cd(site_mag) - natural_cd, 0.0)
    haze = float(np.clip(1.0 + 0.25 * (turbidity - 2.0), 0.75, 1.75))
    profile = 1.0 + 5.0 * np.exp(-np.degrees(alt) / 8.0)
    warmth = np.exp(-np.degrees(alt) / 15.0)[..., None]
    lp_color = (warmth * _unit_luma(_LIGHT_POLLUTION_HORIZON_COLOR)
                + (1.0 - warmth) * _unit_luma(_LIGHT_POLLUTION_ZENITH_COLOR))
    lp = (lp_zenith_cd * haze * profile)[..., None] * lp_color
    return natural, lp


# --- Moonlight ----------------------------------------------------------------
FULL_MOON_ILLUMINANCE_LUX = 0.27
MEAN_MOON_DISTANCE_KM = 384400.0


def moon_illuminance_lux(phase_angle_deg, distance_km):
    """Lunar illuminance at the top of the atmosphere (Krisciunas & Schaefer
    1991 phase law; phase angle 0 = full)."""
    a = abs(phase_angle_deg)
    return (FULL_MOON_ILLUMINANCE_LUX * 10 ** (-0.4 * (0.026 * a + 4e-9 * a ** 4))
            * (MEAN_MOON_DISTANCE_KM / distance_km) ** 2)


# --- Combined sky ---------------------------------------------------------------
def compute_sky(alt_deg, az_deg, sun_alt_deg, sun_az_deg, turbidity, star_visibility,
                moon_alt_deg=None, moon_az_deg=None, moon_phase_angle_deg=None, moon_distance_km=None):
    """Full sky radiance on an alt/az grid. Returns a dict with:
      rgb            linear RGB in cd/m^2 (all sources)
      luminance      cd/m^2 per pixel
      zenith_total   zenith luminance, all sources (exposure reference)
      zenith_dark    zenith luminance from the Sun + natural airglow only --
                     the moonless, unpolluted reference for star visibility."""
    sun_rgb, sun_zenith = _source_sky(alt_deg, az_deg, sun_alt_deg, sun_az_deg,
                                      SOLAR_ILLUMINANCE_LUX, turbidity, twilight_tail=True)
    natural, lp = night_floor(alt_deg, star_visibility, turbidity)
    natural_z, lp_z = night_floor(np.array([90.0]), star_visibility, turbidity)
    rgb = sun_rgb + natural + lp

    moon_zenith = 0.0
    if moon_alt_deg is not None and moon_alt_deg > -2.0:
        e_moon = moon_illuminance_lux(moon_phase_angle_deg, moon_distance_km)
        if e_moon > 1e-4:
            moon_rgb, mz = _source_sky(alt_deg, az_deg, moon_alt_deg, moon_az_deg, e_moon, turbidity,
                                       twilight_tail=False)
            rgb = rgb + moon_rgb
            moon_zenith = luminance(mz)

    zenith_dark = luminance(sun_zenith) + luminance(natural_z[0])
    zenith_total = zenith_dark + luminance(lp_z[0]) + moon_zenith
    return {"rgb": rgb, "luminance": luminance(rgb), "zenith_total": zenith_total, "zenith_dark": zenith_dark}


# --- Display -------------------------------------------------------------------
# Exposure follows the zenith brightness only partially (like an eye or a
# camera that adapts but not completely), so night still reads darker than
# day: displayed level ~ L_ref^(1 - ADAPTATION). Calibrated so a clear
# daytime zenith is a bright blue and a dark moonless night a deep navy.
ADAPTATION = 0.726
EXPOSURE_KEY = 0.026
# Photographic saturation boost (linear, around luminance): a physically
# accurate clear sky maps to a fairly pale sRGB blue, which cameras and the
# eye render more vividly. Stronger with the Sun high (the daytime blue)
# than around sunset, where the colors are already intense.
TWILIGHT_SATURATION = 1.6
DAYTIME_SATURATION = 2.2


def display_saturation(sun_alt_deg):
    w = float(np.clip((sun_alt_deg - 5.0) / 20.0, 0.0, 1.0))
    return TWILIGHT_SATURATION + (DAYTIME_SATURATION - TWILIGHT_SATURATION) * w


def to_display(rgb, zenith_total, saturation=TWILIGHT_SATURATION):
    exposure = EXPOSURE_KEY * zenith_total ** (1.0 - ADAPTATION) / max(zenith_total, 1e-12)
    x = rgb * exposure
    lum = np.maximum(x @ REC709_LUMA, 1e-12)
    mapped = lum / (1.0 + lum)
    out = x * (mapped / lum)[..., None]
    out_lum = (out @ REC709_LUMA)[..., None]
    out = out_lum + saturation * (out - out_lum)
    return np.clip(out, 0.0, 1.0) ** (1.0 / 2.2)


# --- Naked-eye limiting magnitude ---------------------------------------------
def naked_eye_limit(luminance_cd_m2):
    """Naked-eye limiting magnitude for a sky of the given brightness (the
    commonly used SQM -> NELM relation; 21.8 mag/arcsec^2 -> ~6.5)."""
    sqm = surface_brightness_mag(luminance_cd_m2)
    return 7.93 - 5.0 * np.log10(10 ** (4.316 - sqm / 5.0) + 1.0)

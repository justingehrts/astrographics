"""Forecast clouds for the sky graphic.

Data: Open-Meteo's free, keyless forecast API -- hourly low / mid / high
cloud cover (percent) at a fan of points around the observer, so the cloud
amount can differ by direction ("overcast overhead, clear to the west").
One request returns the whole forecast range; the app caches it for hours and
interpolates to the chosen time locally.

Geometry: each layer is a sheet at a typical height. A pixel at altitude a
sees that sheet at a ground distance that grows quickly toward the horizon
(Earth's curvature included), and takes the cover forecast there.

Shapes: the forecast says how much of the sky is covered, not where each
cloud is, so the shapes are procedural noise drawn on the sheet itself (in km).
Perspective then shrinks and stacks cells toward the horizon on its own. The
local cover sets how much of the noise becomes cloud.

Lighting: sunlight and moonlight reaching each layer's height through the
same atmosphere model as the sky (so high clouds stay pink after ground
sunset and cloud bases glow only with a clear path to the Sun), skylight,
city light on cloud bases at night, and haze between the observer and
distant clouds.
"""
from datetime import timezone

import numpy as np
import requests

from astro import atmosphere
from astro.horizon import _destination

FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_S = 15
FORECAST_DAYS = 16
PAST_DAYS = 1

# Sidebar label -> Open-Meteo `models` value (None = Open-Meteo's best-match blend).
MODELS = {
    "Best match (blend)": None,
    "HRRR": "gfs_hrrr",
    "GFS": "gfs_global",  # not gfs_seamless: that blends HRRR in over the US, so it matched HRRR
    "ECMWF": "ecmwf_ifs025",
}

LAYERS = ("low", "mid", "high")
_VARIABLES = ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high")

# Sampling fan: the site, plus rings out to where a high layer meets the horizon.
RAY_AZIMUTHS_DEG = np.arange(0.0, 360.0, 30.0)
RAY_DISTANCES_KM = np.array([10.0, 25.0, 50.0, 100.0, 200.0])

EARTH_RADIUS_KM = 6371.0


# --- Data ------------------------------------------------------------------------
def sample_points(lat, lon):
    """(lats, lons) of the site followed by every (azimuth, distance) point,
    azimuth-major."""
    lats, lons = [lat], [lon]
    for az in RAY_AZIMUTHS_DEG:
        for d in RAY_DISTANCES_KM:
            plat, plon = _destination(lat, lon, az, d)
            lats.append(float(plat))
            lons.append(float(plon))
    return lats, lons


def build_request_params(lat, lon, model_label):
    lats, lons = sample_points(lat, lon)
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "hourly": ",".join(_VARIABLES),
        "timezone": "GMT",
        "timeformat": "unixtime",
        "forecast_days": FORECAST_DAYS,
        "past_days": PAST_DAYS,
    }
    slug = MODELS[model_label]
    if slug:
        params["models"] = slug
    return params


def _series(hourly, var):
    """The hourly values for `var`, accepting the plain key or a model-suffixed
    one (Open-Meteo adds the model name when several are requested)."""
    if var in hourly:
        values = hourly[var]
    else:
        keys = [k for k in hourly if k.startswith(var + "_")]
        if not keys:
            raise KeyError(var)
        values = hourly[keys[0]]
    return np.array([np.nan if v is None else float(v) for v in values])


def parse_response(payload, n_points):
    """Open-Meteo JSON -> {"times": unix seconds (T,), "cover": (3, points, T) in 0-1}."""
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != n_points:
        raise ValueError(f"expected {n_points} locations, got {len(locations)}")
    times = np.array(locations[0]["hourly"]["time"], dtype=float)
    cover = np.empty((len(_VARIABLES), n_points, len(times)))
    for p, loc in enumerate(locations):
        for v, var in enumerate(_VARIABLES):
            cover[v, p] = _series(loc["hourly"], var)
    return {"times": times, "cover": np.clip(cover / 100.0, 0.0, 1.0)}


def fetch_cloud_series(lat, lon, model_label):
    """The whole hourly forecast for the sampling fan. Raises on network or
    parse failure. Callers cache this (hours): it does not depend on the time
    or view, only on the place and model."""
    params = build_request_params(lat, lon, model_label)
    resp = requests.get(FORECAST_API_URL, params=params, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return parse_response(resp.json(), 1 + len(RAY_AZIMUTHS_DEG) * len(RAY_DISTANCES_KM))


def cover_at_time(series, when_utc):
    """Cover grid (3 layers, azimuths, 1 + distances) in 0-1 at `when_utc`
    (a timezone-aware datetime), interpolated between forecast hours. Column 0
    is the site itself. Raises ValueError when the time is outside the forecast."""
    t = when_utc.astimezone(timezone.utc).timestamp()
    times = series["times"]
    if not times[0] <= t <= times[-1]:
        raise ValueError("outside the forecast range")
    i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
    f = (t - times[i]) / (times[i + 1] - times[i])
    values = series["cover"][:, :, i] * (1 - f) + series["cover"][:, :, i + 1] * f
    if np.isnan(values).any():
        raise ValueError("this model has no forecast for that time")
    site = values[:, :1]
    rings = values[:, 1:].reshape(len(_VARIABLES), len(RAY_AZIMUTHS_DEG), len(RAY_DISTANCES_KM))
    return np.concatenate([np.repeat(site[:, :, None], len(RAY_AZIMUTHS_DEG), axis=1), rings], axis=2)


def uniform_cover(low, mid, high):
    """A cover grid with the same amount everywhere (manual mode), 0-1 each."""
    grid = np.empty((3, len(RAY_AZIMUTHS_DEG), 1 + len(RAY_DISTANCES_KM)))
    for i, v in enumerate((low, mid, high)):
        grid[i] = np.clip(v, 0.0, 1.0)
    return grid


def adjust_cover(grid, offsets):
    """Shift each layer's whole field by a number of percentage points
    (forecaster override), clipped to 0-100%."""
    out = grid.copy()
    for i, off in enumerate(offsets):
        out[i] = np.clip(out[i] + off / 100.0, 0.0, 1.0)
    return out


def _cover_lookup(grid_layer, az_deg, dist_km):
    """Bilinear lookup in (azimuth, distance) of one layer's cover grid."""
    dists = np.concatenate([[0.0], RAY_DISTANCES_KM])
    step = RAY_AZIMUTHS_DEG[1] - RAY_AZIMUTHS_DEG[0]
    a = np.mod(az_deg, 360.0) / step
    a0 = np.floor(a).astype(int) % len(RAY_AZIMUTHS_DEG)
    a1 = (a0 + 1) % len(RAY_AZIMUTHS_DEG)
    fa = a - np.floor(a)
    d = np.clip(dist_km, 0.0, dists[-1])
    j = np.clip(np.searchsorted(dists, d) - 1, 0, len(dists) - 2)
    fd = (d - dists[j]) / (dists[j + 1] - dists[j])
    g = grid_layer
    return ((g[a0, j] * (1 - fd) + g[a0, j + 1] * fd) * (1 - fa)
            + (g[a1, j] * (1 - fd) + g[a1, j + 1] * fd) * fa)


# --- Geometry --------------------------------------------------------------------
def sheet_geometry(alt_deg, height_km):
    """For a view altitude, where the line of sight meets a cloud sheet at
    `height_km` above a spherical Earth: (ground distance km, slant range km,
    central angle rad)."""
    a = np.radians(np.maximum(alt_deg, 0.05))
    R = EARTH_RADIUS_KM
    sin_a = np.sin(a)
    slant = -R * sin_a + np.sqrt((R * sin_a) ** 2 + 2 * R * height_km + height_km ** 2)
    theta = np.arcsin(np.clip(slant * np.cos(a) / (R + height_km), -1.0, 1.0))
    return R * theta, slant, theta


# --- Procedural texture ----------------------------------------------------------
def _hash01(ix, iy, seed):
    h = (ix.astype(np.uint64) * np.uint64(0x9E3779B1) + iy.astype(np.uint64) * np.uint64(0x85EBCA77)
         + np.uint64(seed) * np.uint64(0xC2B2AE3D))
    h ^= h >> np.uint64(15)
    h *= np.uint64(0x2C1B3C6D)
    h ^= h >> np.uint64(12)
    h *= np.uint64(0x297A2D39)
    h ^= h >> np.uint64(15)
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def _value_noise(x, y, seed):
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    v00 = _hash01(ix, iy, seed)
    v10 = _hash01(ix + 1, iy, seed)
    v01 = _hash01(ix, iy + 1, seed)
    v11 = _hash01(ix + 1, iy + 1, seed)
    return (v00 * (1 - ux) + v10 * ux) * (1 - uy) + (v01 * (1 - ux) + v11 * ux) * uy


# Per-layer look. `scale_km` = largest feature wavelength; `stretch` > 1 draws
# features long along a seeded wind direction (cirrus streaks); `edge` = how
# crisp cloud edges are at full detail; `thicken` = how much cloud sides hide
# the gaps toward the horizon; `opacity` = optical thickness at full cover;
# `transmit` = share of sunlight a cloud base gets when the Sun is above the
# deck; `lp_gain` = how strongly city light brightens its base at night.
LAYER_STYLE = {
    "low": dict(height_km=1.5, scale_km=0.9, octaves=7, gain=0.52, stretch=1.3, warp=0.45, billow=True,
                edge=0.15, thicken=0.25, opacity=0.97, transmit=0.16, side=0.55, lp_gain=6.0),
    "mid": dict(height_km=4.5, scale_km=1.4, octaves=7, gain=0.55, stretch=1.6, warp=0.35, billow=True,
                edge=0.17, thicken=0.12, opacity=0.92, transmit=0.28, side=0.45, lp_gain=2.5),
    "high": dict(height_km=9.0, scale_km=12.0, octaves=7, gain=0.62, stretch=7.0, warp=0.25, billow=False,
                 edge=0.32, thicken=0.0, opacity=0.35, transmit=0.85, side=0.25, lp_gain=0.6),
}


def _fbm(x, y, footprint_km, style, seed):
    """Fractal noise in [0, 1] with each octave faded out where it is finer
    than the pixel footprint (anti-aliasing: distant clouds blur into a
    band instead of sparkling). Returns (noise, retained detail 0-1)."""
    total = np.zeros_like(x)
    norm = 0.0
    kept = np.zeros_like(x)
    kept_norm = 0.0
    coarse = None
    amp = 1.0
    wavelength = style["scale_km"]
    for k in range(style["octaves"]):
        if k == 3:
            coarse = total / norm  # the large-scale shape alone (smooth): used for shading
        n = _value_noise(x / wavelength, y / wavelength, seed + 101 * k)
        if style["billow"] and k > 0:
            # Folded noise: rounded puffs with sharp creases between them.
            n = 1.0 - np.abs(2.0 * n - 1.0)
        w = np.clip(wavelength / (2.0 * np.maximum(footprint_km, 1e-6)) - 1.0, 0.0, 1.0)
        total += amp * (w * n + (1 - w) * 0.5)
        kept += (amp * w) ** 2
        norm += amp
        kept_norm += amp ** 2
        amp *= style["gain"]
        wavelength /= 2.0
    return total / norm, np.sqrt(kept / kept_norm), coarse


_CDF_CACHE = {}


def _uniformize(n, layer, style, seed):
    """Map full-detail noise values to a uniform 0-1 scale, so a cover of c
    turns a fraction c of the sheet into cloud."""
    if layer not in _CDF_CACHE:
        rng = np.random.default_rng(12345)
        sx, sy = rng.uniform(-500, 500, (2, 60000))
        ref, _, _ = _fbm(sx, sy, np.full_like(sx, 1e-6), style, seed)
        _CDF_CACHE[layer] = np.quantile(ref, np.linspace(0, 1, 513))
    q = _CDF_CACHE[layer]
    return np.interp(n, q, np.linspace(0, 1, len(q)))


def _layer_alpha(layer, cover_grid, az, alt, d_km, footprint_km, seed):
    style = LAYER_STYLE[layer]
    rng = np.random.default_rng(seed + 7)
    wind = rng.uniform(0, np.pi)
    x, y = d_km * np.sin(np.radians(az)), d_km * np.cos(np.radians(az))
    # Rotate into the wind frame and stretch along it.
    u = (x * np.cos(wind) + y * np.sin(wind)) / style["stretch"]
    v = -x * np.sin(wind) + y * np.cos(wind)
    fp = footprint_km / np.sqrt(style["stretch"])
    # Domain warp: lumpier, less grid-like cells.
    wx = _value_noise(u / (style["scale_km"] * 0.7), v / (style["scale_km"] * 0.7), seed + 911) - 0.5
    wy = _value_noise(u / (style["scale_km"] * 0.7) + 17.3, v / (style["scale_km"] * 0.7), seed + 912) - 0.5
    u = u + wx * style["warp"] * style["scale_km"]
    v = v + wy * style["warp"] * style["scale_km"]
    n, detail, coarse = _fbm(u, v, fp, style, seed)
    n = _uniformize(n, layer, style, seed)

    c = _cover_lookup(cover_grid, az, d_km)
    # Toward the horizon cloud sides hide the gaps between cumuliform clouds.
    tan_a = np.tan(np.radians(np.maximum(alt, 0.5)))
    c = 1.0 - (1.0 - c) ** (1.0 + style["thicken"] / tan_a)
    # Crisp edges at full detail; where detail is filtered away, the expected
    # opacity tends to the cover itself.
    width = style["edge"] + (1.0 - style["edge"]) * (1.0 - detail)
    ramp = np.clip((n - (1.0 - c)) / width + 0.5, 0.0, 1.0)
    # Smoothstep: rims fade in gradually rather than ending at a hard line.
    # Symmetric about the threshold, so the mean coverage is unchanged.
    alpha = ramp * ramp * (3.0 - 2.0 * ramp)
    alpha = np.where(c <= 0.0, 0.0, alpha)
    # How deep inside the cloud this point is (0 at a thin edge, 1 in a thick
    # core); where detail is filtered away, use the cover as the average.
    # Smooth: from the large-scale shape only, so decks show soft light and dark
    # patches rather than contour lines that follow the fine texture.
    thick = np.clip(c + (coarse - 0.5) * 1.6, 0.0, 1.0)
    thick = detail * thick + (1.0 - detail) * c
    if layer == "high":
        # Thin ice cloud: optical depth grows along the slant path.
        opacity = 1.0 - np.exp(-style["opacity"] / np.maximum(np.sin(np.radians(np.maximum(alt, 0.5))), 0.05))
    else:
        opacity = style["opacity"]
    return alpha * opacity, thick


# --- Lighting ------------------------------------------------------------------
def _hg(cos_t, g):
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * cos_t) ** 1.5)


def _source_light(view_dirs, point_up, height_m, src_alt, src_az, illuminance, style, turbidity, thick):
    """Cloud luminance (linear RGB, cd/m^2) from one distant light source."""
    src = atmosphere.unit_vectors(np.array(src_alt, float), np.array(src_az, float))
    mu = point_up @ src
    trans = atmosphere.transmittance_to_light(height_m, mu, turbidity)
    thin = 1.0 - thick
    # Light under the deck (the Sun below the cloud's own horizontal, but not yet
    # in the Earth's shadow) strikes the bases directly: sunset-lit clouds, with
    # the thin parts glowing most.
    under = np.clip((0.03 - mu) / 0.06, 0.0, 1.0)
    lit_base = 0.35 + 0.45 * thin
    # Sun above the deck: bases get what diffuses through (less in thick cores),
    # sides and tops in view are lit directly, and thin edges near the Sun glow
    # (forward scattering: the silver lining).
    up = np.clip(mu, 0.0, 1.0)
    through = style["transmit"] * (0.4 + 0.6 * up) * (0.55 + 0.6 * thin)
    sides = style["side"] * (0.25 + 0.75 * np.sqrt(up)) * (0.45 + 0.55 * thin)
    cos_t = view_dirs @ src
    silver = 1.2 * style["transmit"] * _hg(cos_t, 0.8) * 4 * np.pi * 0.05 * thin
    factor = under * lit_base + (1 - under) * (through + sides + silver)
    return illuminance * trans * factor[..., None] / np.pi


def render_clouds(cover_grid, az_min, az_max, alt_max, width, height, sun_alt, sun_az, turbidity,
                  star_visibility, sky_rgb, sky_zenith_rgb, seed,
                  moon_alt=None, moon_az=None, moon_illuminance=0.0):
    """Clouds over the view on a (height, width) grid. `sky_rgb` is the clear
    sky's linear RGB on that grid (for haze in front of distant clouds) and
    `sky_zenith_rgb` the zenith sky (skylight on cloud bases). Returns
    (premultiplied-free linear RGB of the clouds, total opacity 0-1, mean
    cover per layer at the site)."""
    az = np.linspace(az_min, az_max, width)
    alt = np.linspace(0.0, alt_max, height)
    AZ, ALT = np.meshgrid(az, alt)
    view_dirs = atmosphere.unit_vectors(ALT, AZ)
    d_alt_rad = np.radians(alt_max / max(height - 1, 1))
    d_az_rad = np.radians((az_max - az_min) / max(width - 1, 1))

    natural_z, lp_z = atmosphere.night_floor(np.array([90.0]), star_visibility, turbidity)
    lp_level = float(atmosphere.luminance(lp_z[0]))
    lp_color = np.array([1.0, 0.80, 0.58])
    lp_color = lp_color / (lp_color @ atmosphere.REC709_LUMA)
    haze_km = 120.0 / max(turbidity, 1.0)

    premult = np.zeros(ALT.shape + (3,))
    total_alpha = np.zeros(ALT.shape)
    for li, layer in reversed(list(enumerate(LAYERS))):  # far (high) to near (low)
        style = LAYER_STYLE[layer]
        if cover_grid[li].max() <= 0.0:
            continue
        d_km, slant_km, theta = sheet_geometry(ALT, style["height_km"])
        d_up, _, _ = sheet_geometry(np.minimum(ALT + 0.05, 90.0), style["height_km"])
        dd_dalt = np.abs(d_km - d_up) / np.radians(0.05)
        footprint = np.maximum(dd_dalt * d_alt_rad, d_km * d_az_rad)
        alpha, thick = _layer_alpha(layer, cover_grid[li], AZ, ALT, d_km, footprint, seed + 1000 * li)
        if not alpha.any():
            continue

        # Local vertical at the cloud point (observer-centered, Earth-centered frame).
        az_r = np.radians(AZ)
        point_up = np.stack([np.sin(theta) * np.sin(az_r), np.sin(theta) * np.cos(az_r), np.cos(theta)], axis=-1)
        h_m = style["height_km"] * 1000.0
        light = _source_light(view_dirs, point_up, h_m, sun_alt, sun_az, atmosphere.SOLAR_ILLUMINANCE_LUX,
                              style, turbidity, thick)
        if moon_alt is not None and moon_illuminance > 0:
            light = light + _source_light(view_dirs, point_up, h_m, moon_alt, moon_az, moon_illuminance,
                                          style, turbidity, thick)
        shade = (0.6 + 0.4 * (1.0 - thick))[..., None]  # thick cores are darker underneath
        light = light + (0.55 * sky_zenith_rgb + 0.25 * sky_rgb) * shade  # skylight on the base
        light = light + style["lp_gain"] * lp_level * lp_color * shade  # city light on the base

        # Haze between the observer and the cloud fades it into the sky.
        t_view = np.exp(-slant_km / haze_km)[..., None]
        color = light * t_view + sky_rgb * (1 - t_view)

        a = alpha[..., None]
        premult = color * a + premult * (1 - a)
        total_alpha = alpha + total_alpha * (1 - alpha)

    rgb = premult / np.maximum(total_alpha, 1e-6)[..., None]
    site_cover = cover_grid[:, 0, 0]
    return rgb, total_alpha, site_cover

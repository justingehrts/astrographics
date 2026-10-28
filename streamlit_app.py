import logging
import time as time_module
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")  # Safe headless execution for cloud servers
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import io
import pathlib
import numpy as np

# Core astronomical math engine
from skyfield.api import wgs84, Star, Loader
from skyfield.data import hipparcos
from skyfield.magnitudelib import planetary_magnitude

from astro import asterisms, atmosphere, clouds, constellations, extinction, horizon, location, moon, stars, treeline

logger = logging.getLogger(__name__)

# Standard-atmosphere constants used to enable refraction for all altaz()
# calls below. This app has no live local weather input, so these are a
# reasonable fixed approximation -- still meaningfully more accurate near
# the horizon (this app's whole 0-40+deg window) than ignoring refraction
# entirely, which is what happened before.
STANDARD_TEMPERATURE_C = 10.0
STANDARD_PRESSURE_MBAR = 1010.0

# Image formats. 16:9 is the default; the others are for social media. A degree
# takes the same space horizontally and vertically (no stretching of trees,
# houses or the Moon) when the altitude shown is the field-of-view width times
# height/width. Every format's figure has the same area (so fonts, lines and
# markers read alike) and is saved at the pixel size its platforms expect.
FIGURE_AREA_IN2 = 81.0  # 12 x 6.75 in, the original 16:9 figure
FORMATS = {
    "16x9": dict(label="16:9 landscape (default)", aspect=(16, 9), export_px=(1920, 1080), default_fov=80.0,
                 safe_zone=None),
    "9x16": dict(label="9:16 vertical (Stories, Reels, TikTok, Shorts)", aspect=(9, 16), export_px=(1080, 1920),
                 default_fov=45.0, safe_zone=dict(top=0.14, bottom=0.20, side=0.0)),
    "4x5": dict(label="4:5 portrait (Instagram, Facebook feed)", aspect=(4, 5), export_px=(1080, 1350),
                default_fov=60.0, safe_zone=dict(top=0.05, bottom=0.05, side=0.05)),
    "1x1": dict(label="1:1 square (any feed)", aspect=(1, 1), export_px=(1080, 1080), default_fov=70.0,
                safe_zone=None),
}
DEFAULT_FORMAT = "16x9"
ALT_MAX_RANGE = (20.0, 90.0)
ALT_MAX_STEP = 5.0
FOV_RANGE = (30.0, 180.0)
FOV_STEP = 5.0


def figure_size_in(fmt):
    """(width, height) in inches of a format's figure: same area for all."""
    aw, ah = FORMATS[fmt]["aspect"]
    width = (FIGURE_AREA_IN2 * aw / ah) ** 0.5
    return width, FIGURE_AREA_IN2 / width


def export_dpi(fmt):
    return FORMATS[fmt]["export_px"][0] / figure_size_in(fmt)[0]


def natural_altitude_for_fov(fov_deg, fmt=DEFAULT_FORMAT):
    """Max altitude that gives equal-angle proportions for `fov_deg` in a
    format, rounded to the altitude slider's step and limited to its range."""
    fig_w, fig_h = figure_size_in(fmt)
    ideal = fov_deg * fig_h / fig_w
    stepped = np.floor(ideal / ALT_MAX_STEP + 0.5) * ALT_MAX_STEP
    return float(np.clip(stepped, *ALT_MAX_RANGE))


def max_fov_for_lock(fmt):
    """Widest view whose natural altitude still fits the altitude limit
    (vertical formats cannot be wide: 9:16 tops out near 50 degrees)."""
    fig_w, fig_h = figure_size_in(fmt)
    widest = ALT_MAX_RANGE[1] * fig_w / fig_h
    return float(np.clip(np.floor(widest / FOV_STEP) * FOV_STEP, FOV_RANGE[0], FOV_RANGE[1]))


# Figure size in inches; set from the chosen format in the sidebar (below).
FIG_WIDTH_IN, FIG_HEIGHT_IN = figure_size_in(DEFAULT_FORMAT)

# Set page layout to wide for a clean dashboard feel
st.set_page_config(layout="wide", page_title="Custom Sky Graphic Generator")

st.title("🌌 Broadcast Sky Graphic Generator")
st.write("A lightweight, reliable engine rendering clean astronomical plates with native horizon silhouettes.")

# ======================================================================
# CACHED DATA ENGINE
# Loads the heavy ephemeris and star catalog files into memory exactly once,
# preventing timeouts and massive latency delays on cloud deployments.
# ======================================================================
# The catalog ships with the repo (a gzipped copy of the ESA Hipparcos main
# catalog, CDS I/239) so a cold start never depends on a download. The URLs are
# only a fallback if that file is missing.
HIPPARCOS_LOCAL = pathlib.Path(__file__).parent / "data" / "hip_main.dat.gz"
HIPPARCOS_URLS = (hipparcos.URL, "https://vizier.cfa.harvard.edu/ftp/cats/I/239/hip_main.dat",
                  "https://cdsarc.u-strasbg.fr/ftp/cats/I/239/hip_main.dat")


def open_hipparcos(loader, attempts=3):
    """Open the Hipparcos catalog, downloading it if this container has no copy.
    The catalog host is occasionally unreachable, so try each known mirror a few
    times before giving up with a readable message (instead of a raw traceback)."""
    if HIPPARCOS_LOCAL.exists():
        return open(HIPPARCOS_LOCAL, "rb")
    last_error = None
    for url in HIPPARCOS_URLS:
        for attempt in range(attempts):
            try:
                return loader.open(url, filename="hip_main.dat")
            except OSError as exc:
                last_error = exc
                time_module.sleep(1.5 * (attempt + 1))
    st.error("The star catalog could not be downloaded (its server may be down). "
             "Please try again in a few minutes.")
    st.caption(f"Last error: {last_error}")
    st.stop()


@st.cache_resource
def get_astronomy_data():
    loader = Loader('.')
    ts = loader.timescale()
    # The planetary ephemeris (JPL DE421) ships in data/; Skyfield only downloads
    # it from NASA/JPL if that file is missing.
    eph = Loader(str(pathlib.Path(__file__).parent / "data"))('de421.bsp')

    # Forces Streamlit Cloud to download the catalog natively if missing
    with open_hipparcos(loader) as f:
        stars_df = hipparcos.load_dataframe(f)
        bv = stars.load_bv_series(f)

    # Many fainter Hipparcos entries have an incomplete astrometric
    # solution (blank parallax/proper-motion fields in the raw catalog,
    # which hipparcos.load_dataframe parses as NaN). Skyfield's vectorized
    # position/light-time calculations don't tolerate NaN inputs -- one
    # such star pulled in by a high enough magnitude cutoff corrupts the
    # whole batch and crashes deep in Skyfield's relativistic deflection
    # code. Drop those rows up front rather than downstream of the crash.
    stars_df = stars_df.dropna(subset=[
        "ra_degrees", "dec_degrees", "parallax_mas",
        "ra_mas_per_year", "dec_mas_per_year", "magnitude",
    ])
    stars_df = stars_df.join(bv)

    return ts, eph, stars_df


@st.cache_resource
def get_constellation_segments():
    return constellations.load_constellation_segments()


@st.cache_resource
def get_horizon_strips(scene, leaves):
    return treeline.load_scene_strips(scene, leaves=leaves)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_sky_defaults(lat, lon):
    return location.suggest_sky_defaults(lat, lon)


@st.cache_data(ttl=3600, show_spinner="Sampling real terrain horizon...")
def cached_horizon_profile(lat, lon, az_min, az_max, n_samples=96):
    sample_az = np.linspace(az_min, az_max, n_samples)
    return sample_az, horizon.fetch_horizon_profile(lat, lon, sample_az)


SKY_GRID_W, SKY_GRID_H = 150, 100


@st.cache_data(max_entries=64, show_spinner=False)
def cached_sky(az_min, az_max, alt_max, sun_alt, sun_az, turbidity, star_visibility,
               moon_alt_deg=None, moon_az_deg=None, moon_phase_angle_deg=None, moon_distance_km=None):
    """Sky background for the view: (display RGB image, luminance in cd/m^2
    per pixel, moonless/unpolluted zenith luminance for star visibility)."""
    X, Y = np.meshgrid(np.linspace(az_min, az_max, SKY_GRID_W), np.linspace(0, alt_max, SKY_GRID_H))
    sky = atmosphere.compute_sky(
        Y, X, sun_alt, sun_az, turbidity, star_visibility,
        moon_alt_deg=moon_alt_deg, moon_az_deg=moon_az_deg,
        moon_phase_angle_deg=moon_phase_angle_deg, moon_distance_km=moon_distance_km,
    )
    display = atmosphere.to_display(sky["rgb"], sky["zenith_total"], atmosphere.display_saturation(sun_alt))
    return display, sky["luminance"], sky["zenith_dark"], sky["rgb"], sky["zenith_total"]


# Forecast clouds: the whole hourly series for a place and model, kept for
# 3 hours. Time, date and view changes reuse it (interpolated locally); only a
# new place, a new model or the expiry calls Open-Meteo again.
@st.cache_data(ttl=3 * 3600, show_spinner="Fetching forecast clouds...")
def _cloud_series_3h(lat_rounded, lon_rounded, model_label):
    return clouds.fetch_cloud_series(lat_rounded, lon_rounded, model_label)


@st.cache_data(ttl=600, show_spinner=False)
def cached_cloud_series(lat_rounded, lon_rounded, model_label):
    """The 3-hour cached series, or None if the fetch failed. Streamlit never
    caches an exception, so without this a failing model would be requested
    again on every rerun; a failure is remembered for 10 minutes instead."""
    try:
        return _cloud_series_3h(lat_rounded, lon_rounded, model_label)
    except Exception:
        logger.exception("Cloud forecast fetch failed for %s", model_label)
        return None


CLOUD_GRID_LONG_SIDE = 768  # pixels on the longer side; the other follows the image format


def cloud_grid_size(fmt):
    """(width, height) of the cloud layer's grid, in the image format's shape."""
    aw, ah = FORMATS[fmt]["aspect"]
    scale = CLOUD_GRID_LONG_SIDE / max(aw, ah)
    return int(round(aw * scale)), int(round(ah * scale))


def _upsample_bilinear(img, height, width):
    """Bilinear resize of an (h, w, channels) array (both grids span the same
    corners), in numpy only."""
    h, w = img.shape[:2]
    ys, xs = np.linspace(0, h - 1, height), np.linspace(0, w - 1, width)
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
    fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = img[y0][:, x0] * (1 - fx) + img[y0][:, x0 + 1] * fx
    bottom = img[y0 + 1][:, x0] * (1 - fx) + img[y0 + 1][:, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


@st.cache_data(max_entries=16, show_spinner=False)
def cached_clouds(cover_grid, az_min, az_max, alt_max, sun_alt, sun_az, turbidity, star_visibility,
                  sky_rgb_coarse, seed, grid_w, grid_h, moon_alt=None, moon_az=None, moon_illuminance=0.0, **moon_sky_kwargs):
    """Cloud layer for the view as display RGBA (clouds composited over nothing,
    with their own opacity), on a grid in the image format's shape. Lit and exposed exactly like the
    sky so the two match."""
    sky_rgb = _upsample_bilinear(sky_rgb_coarse, grid_h, grid_w)
    zenith = atmosphere.compute_sky(np.array([[90.0]]), np.array([[0.0]]), sun_alt, sun_az, turbidity,
                                    star_visibility, **moon_sky_kwargs)
    rgb, alpha, _ = clouds.render_clouds(
        cover_grid, az_min, az_max, alt_max, grid_w, grid_h, sun_alt, sun_az, turbidity,
        star_visibility, sky_rgb, zenith["rgb"][0, 0], seed,
        moon_alt=moon_alt, moon_az=moon_az, moon_illuminance=moon_illuminance,
    )
    display = atmosphere.to_display(rgb, zenith["zenith_total"], atmosphere.display_saturation(sun_alt))
    return np.dstack([display, alpha])


# Load the data models natively
ts, eph, stars_df = get_astronomy_data()
CONSTELLATION_SEGMENTS = get_constellation_segments()
earth = eph['earth']
sun = eph['sun']


def _seed_session_state_once(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def normalize_az(az, az_min, az_max):
    """Shifts a raw 0-360deg azimuth by +-360 when needed so it lands
    inside [az_min, az_max] -- the view window can extend outside [0,360)
    (e.g. bearing=10, FOV=60 gives az_min=-20) or past it (bearing=350
    gives az_max=380), and Skyfield always returns azimuths in [0,360).
    Works for both scalar and array `az`."""
    az = np.asarray(az, dtype=float)
    up, down = az + 360.0, az - 360.0
    in_window = (az >= az_min) & (az <= az_max)
    use_up = ~in_window & (up >= az_min) & (up <= az_max)
    use_down = ~in_window & ~use_up & (down >= az_min) & (down <= az_max)
    result = np.where(use_up, up, az)
    result = np.where(use_down, down, result)
    return result


COMPASS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def compass_ticks(az_min, az_max):
    """Returns [(az, label), ...] for each of the 16 compass points that
    falls inside [az_min, az_max], in whichever +-360deg-shifted form
    actually lands in that window (same wraparound az_min/az_max can
    have as normalize_az handles for plotted bodies)."""
    start_k = int(np.floor(az_min / 22.5)) - 1
    end_k = int(np.ceil(az_max / 22.5)) + 1
    ticks = []
    for k in range(start_k, end_k + 1):
        az = k * 22.5
        if az_min <= az <= az_max:
            ticks.append((az, COMPASS_16[k % 16]))
    return ticks


# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Observation Settings")

try:
    url_date = datetime.strptime(st.query_params.get("date", ""), "%Y-%m-%d").date()
except ValueError:
    url_date = datetime.now().date()
obs_date = st.sidebar.date_input("Select Date", url_date)

# Generate clean 12-hour time strings (AM/PM) with naive time object values
time_options = []
time_labels = []

for hour in range(24):
    for minute in [0, 15, 30, 45]:
        t_obj = time(hour, minute)
        time_options.append(t_obj)

        ampm_label = t_obj.strftime("%I:%M %p")
        time_labels.append(ampm_label)

# Restore time from the URL if present, else default to 09:15 PM
try:
    url_time = datetime.strptime(st.query_params.get("time", "21:15"), "%H:%M").time()
    default_time_index = time_options.index(url_time) if url_time in time_options else time_labels.index("09:15 PM")
except ValueError:
    default_time_index = time_labels.index("09:15 PM") if "09:15 PM" in time_labels else 0

obs_time = st.sidebar.selectbox(
    "Select Time",
    options=time_options,
    format_func=lambda x: x.strftime("%I:%M %p"),
    index=default_time_index
)

tz_options = ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
              "America/Phoenix", "America/Anchorage", "Pacific/Honolulu"]
url_tz = st.query_params.get("tz", tz_options[0])
selected_tz = st.sidebar.selectbox("Time Zone", tz_options, index=tz_options.index(url_tz) if url_tz in tz_options else 0)


def describe_local_time(day, clock, tz_name):
    """(label, warning) for a wall-clock time: the zone abbreviation and UTC offset
    in effect on that date (EDT vs EST), plus a note when the time falls in the
    hour skipped or repeated by a daylight-saving change."""
    zone = ZoneInfo(tz_name)
    first = datetime.combine(day, clock, tzinfo=zone)
    label = f"{first.tzname()} (UTC{first.utcoffset().total_seconds() / 3600:+g})"
    roundtrip = first.astimezone(ZoneInfo("UTC")).astimezone(zone)
    if roundtrip.replace(tzinfo=None) != first.replace(tzinfo=None):
        # Skipped hour (spring forward): the clock jumps over this time.
        return label, f"{clock:%I:%M %p} does not exist today (clocks spring forward); using {roundtrip:%I:%M %p} {roundtrip.tzname()}."
    if first.utcoffset() != first.replace(fold=1).utcoffset():
        # Repeated hour (fall back): the earlier, daylight-time reading is used.
        return label, f"{clock:%I:%M %p} happens twice today; using the first ({first.tzname()})."
    return label, None


_tz_label, _tz_warning = describe_local_time(obs_date, obs_time, selected_tz)
st.sidebar.caption(f"Times are local: {_tz_label}")
if _tz_warning:
    st.sidebar.warning(_tz_warning)

# ======================================================================
# URL QUERY PARAMETER ENGINE
# ======================================================================
try:
    url_lat = np.clip(float(st.query_params.get("lat", 39.96)), -90.0, 90.0)
except ValueError:
    url_lat = 39.96

try:
    url_lon = np.clip(float(st.query_params.get("lon", -83.00)), -180.0, 180.0)
except ValueError:
    url_lon = -83.00

lat = st.sidebar.number_input("Latitude", value=url_lat, step=0.01, format="%.2f", min_value=-90.0, max_value=90.0)
lon = st.sidebar.number_input("Longitude", value=url_lon, step=0.01, format="%.2f", min_value=-180.0, max_value=180.0)

st.sidebar.header("2. View Window")

try:
    url_bearing = np.clip(float(st.query_params.get("bearing", 225.0)), 0.0, 359.0)
except ValueError:
    url_bearing = 225.0
_url_fmt = st.query_params.get("fmt", DEFAULT_FORMAT)
url_fmt = _url_fmt if _url_fmt in FORMATS else DEFAULT_FORMAT
try:
    url_fov = np.clip(float(st.query_params.get("fov", FORMATS[url_fmt]["default_fov"])), *FOV_RANGE)
except ValueError:
    url_fov = FORMATS[url_fmt]["default_fov"]
try:
    url_altmax = np.clip(float(st.query_params.get("altmax", 45.0)), 20.0, 90.0)
except ValueError:
    url_altmax = 45.0

# Lock altitude to natural proportions: on for a fresh visit. A link made
# before this option existed carries an explicit altitude and no "lock"
# value; keep that view exactly as it was saved.
_url_lock = st.query_params.get("lock")
if _url_lock is not None:
    url_lock = _url_lock == "1"
else:
    url_lock = "altmax" not in st.query_params

DIRECTION_PRESETS = {
    "Custom": None, "North": 0.0, "Northeast": 45.0, "East": 90.0, "Southeast": 135.0,
    "South": 180.0, "Southwest": 225.0, "West": 270.0, "Northwest": 315.0,
}


def _apply_direction_preset():
    preset = DIRECTION_PRESETS[st.session_state["direction_preset"]]
    if preset is not None:
        st.session_state["bearing"] = preset


def _clear_preset_if_bearing_diverged():
    preset = DIRECTION_PRESETS[st.session_state.get("direction_preset", "Custom")]
    if preset is not None and st.session_state["bearing"] != preset:
        # The bearing no longer matches the selected preset (the user
        # nudged the slider directly), so the dropdown's label would
        # otherwise be left showing a stale/misleading direction name.
        st.session_state["direction_preset"] = "Custom"


_seed_session_state_once("direction_preset", "Custom")
_seed_session_state_once("bearing", url_bearing)
_seed_session_state_once("fov_width", url_fov)
_seed_session_state_once("alt_max", url_altmax)
_seed_session_state_once("lock_proportions", url_lock)
_seed_session_state_once("image_format", url_fmt)


def _apply_format_defaults():
    """Picking a format starts from that format's natural view width."""
    st.session_state["fov_width"] = FORMATS[st.session_state["image_format"]]["default_fov"]

st.sidebar.selectbox(
    "Direction Preset", list(DIRECTION_PRESETS.keys()),
    key="direction_preset", on_change=_apply_direction_preset,
)
bearing = st.sidebar.slider(
    "Compass Bearing (0=N, 90=E, 180=S, 270=W)", 0.0, 359.0, step=1.0,
    key="bearing", on_change=_clear_preset_if_bearing_diverged,
)
image_format = st.sidebar.selectbox(
    "Image Format", list(FORMATS), format_func=lambda k: FORMATS[k]["label"], key="image_format",
    on_change=_apply_format_defaults,
    help="16:9 for broadcast. The others are sized for social media; picking one starts from a "
         "view width that suits its shape.",
)
FIG_WIDTH_IN, FIG_HEIGHT_IN = figure_size_in(image_format)
# With the altitude locked to natural proportions, a tall frame cannot show a
# wide view (the altitude would pass 90 degrees), so cap the width slider there.
fov_max = max_fov_for_lock(image_format) if st.session_state["lock_proportions"] else FOV_RANGE[1]
if st.session_state["fov_width"] > fov_max:
    st.session_state["fov_width"] = fov_max
fov_width = st.sidebar.slider("Field of View Width (deg)", FOV_RANGE[0], fov_max, step=FOV_STEP, key="fov_width")
lock_proportions = st.sidebar.checkbox(
    "Lock altitude to natural proportions", key="lock_proportions",
    help="Keeps the altitude shown matched to the width (at the chosen image shape) so a "
         "degree looks the same size sideways and upward -- trees, houses, and the Moon aren't "
         "stretched. Untick to set the altitude yourself.",
)
if lock_proportions:
    # Set before the slider below is created (Streamlit only allows that order).
    st.session_state["alt_max"] = natural_altitude_for_fov(fov_width, image_format)
alt_max = st.sidebar.slider("Max Altitude Shown (deg)", ALT_MAX_RANGE[0], ALT_MAX_RANGE[1],
                            step=ALT_MAX_STEP, key="alt_max", disabled=lock_proportions)

az_min = bearing - fov_width / 2.0
az_max = bearing + fov_width / 2.0

# --- SIDEBAR: 3. SKY CONDITIONS ---
st.sidebar.header("3. Sky Conditions")

# Location-aware starting point for turbidity/star-brightness: on first
# load (unless a shared link already specified them) or whenever the
# location changes, suggest a default from a free reverse-geocoding
# heuristic (see astro/location.py for why there's no true light-pollution
# API involved). The sliders stay fully user-adjustable afterward -- this
# only seeds st.session_state, which the slider widgets below read as
# their starting value via `key=`.
_loc_key = (round(lat, 2), round(lon, 2))
_prev_loc_key = st.session_state.get("_sky_defaults_loc_key")
_had_url_sky_params = "turbidity" in st.query_params or "brightness" in st.query_params

if _had_url_sky_params:
    if "turbidity" not in st.session_state:
        try:
            st.session_state["turbidity"] = np.clip(float(st.query_params["turbidity"]), 1.0, 5.0)
        except (KeyError, ValueError):
            pass
    if "star_brightness" not in st.session_state:
        try:
            st.session_state["star_brightness"] = np.clip(float(st.query_params["brightness"]), 1.0, 4.5)
        except (KeyError, ValueError):
            pass

_should_suggest_defaults = (_prev_loc_key is not None and _prev_loc_key != _loc_key) or (
    _prev_loc_key is None and "turbidity" not in st.session_state
)
if _should_suggest_defaults:
    st.session_state["turbidity"], st.session_state["star_brightness"] = cached_sky_defaults(lat, lon)
st.session_state["_sky_defaults_loc_key"] = _loc_key

# Safety net for a partial/malformed shared link (e.g. only "turbidity" set,
# not "brightness"): guarantee both keys exist before the sliders below are
# created, since they no longer pass a `value=` fallback of their own.
_seed_session_state_once("turbidity", location.SUBURBAN_DEFAULT[0])
_seed_session_state_once("star_brightness", location.SUBURBAN_DEFAULT[1])

turbidity = st.sidebar.slider("Atmospheric Haze (Turbidity)", 1.0, 5.0, step=0.5, key="turbidity")
star_brightness = st.sidebar.slider("Star Visibility Limit", 1.0, 4.5, step=0.5, key="star_brightness")

sky_conditions = {
    1.0: "🏙️ Heavy City Light Pollution (Only exceptionally bright anchor stars appear)",
    1.5: "🌆 Urban Sky (Only major stars like Vega, Capella, or Arcturus are visible)",
    2.0: "🏘️ Bright Suburban Sky (Standard neighborhood viewing conditions; Polaris visible)",
    2.5: "🏡 Typical Suburban Sky (Shows the primary stars people can spot from backyards)",
    3.0: "🌳 Dark Suburban / Rural Fringe (Fainter structural stars begin to show)",
    3.5: "🚜 Rural Country Sky (Excellent visibility; traces out full constellation stick figures)",
    4.0: "🌌 Very Dark Sky (Highly detailed star field; great for deep space tracking)",
    4.5: "✨ Pristine Dark Sky / Desert Void (Maximum density; can clutter a broadcast graphic)"
}

st.sidebar.caption(f"**Current Viewport Simulation:** \n{sky_conditions[star_brightness]}")

# Tree foliage for the horizon scene. Auto follows the observation date (leaves
# April-October, bare the rest of the year, flipped in the Southern Hemisphere);
# the other two override it. A shared link carries the choice, not the season.
FOLIAGE_OPTIONS = {"auto": "Auto (by date)", "on": "Leaves", "off": "Bare (winter)"}
_url_leaves = st.query_params.get("leaves", "auto")
_seed_session_state_once("foliage", _url_leaves if _url_leaves in FOLIAGE_OPTIONS else "auto")
foliage = st.sidebar.selectbox(
    "Tree Foliage", list(FOLIAGE_OPTIONS), format_func=FOLIAGE_OPTIONS.get, key="foliage",
    help="Whether deciduous trees on the horizon have leaves. Auto uses the date selected above: "
         "leaves April-October, bare November-March (reversed south of the equator).",
)
season_has_leaves = treeline.foliage_season_default(obs_date, lat)
horizon_has_leaves = season_has_leaves if foliage == "auto" else foliage == "on"
if foliage == "auto":
    st.sidebar.caption(f"Auto: {'leaves' if season_has_leaves else 'bare trees'} for this date and latitude.")

# Forecast clouds (off by default, so astronomy-event graphics stay clear).
# Forecast mode draws the model's low/mid/high cover around the site, with
# forecaster adjustments in percentage points; Manual sets the amounts directly.
CLOUD_MODES = {"off": "Off", "forecast": "Forecast", "manual": "Manual"}
CLOUD_MODEL_KEYS = {"best": "Best match (blend)", "hrrr": "HRRR", "gfs": "GFS", "ecmwf": "ECMWF"}


def _url_triplet(name, default, lo, hi):
    try:
        values = [float(v) for v in st.query_params.get(name, "").split(",")]
        if len(values) == 3:
            return [float(np.clip(v, lo, hi)) for v in values]
    except ValueError:
        pass
    return list(default)


_url_clouds = st.query_params.get("clouds", "off")
_seed_session_state_once("clouds_mode", _url_clouds if _url_clouds in CLOUD_MODES else "off")
_url_cmodel = st.query_params.get("cmodel", "best")
_seed_session_state_once("cloud_model", _url_cmodel if _url_cmodel in CLOUD_MODEL_KEYS else "best")
for _key, _v in zip(("cloud_adj_low", "cloud_adj_mid", "cloud_adj_high"), _url_triplet("cadj", (0, 0, 0), -100, 100)):
    _seed_session_state_once(_key, _v)
for _key, _v in zip(("cloud_man_low", "cloud_man_mid", "cloud_man_high"), _url_triplet("cman", (30, 20, 20), 0, 100)):
    _seed_session_state_once(_key, _v)

clouds_mode = st.sidebar.selectbox(
    "Clouds", list(CLOUD_MODES), format_func=CLOUD_MODES.get, key="clouds_mode",
    help="Forecast draws the forecast model's low, mid and high cloud cover around your location at the "
         "selected time (cloud shapes are illustrative; amounts and directions follow the forecast). "
         "Manual sets the amounts yourself.",
)
cloud_cover_grid = None
if clouds_mode == "forecast":
    cloud_model = st.sidebar.selectbox("Forecast Model", list(CLOUD_MODEL_KEYS), format_func=CLOUD_MODEL_KEYS.get,
                                       key="cloud_model")
    _when = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
    # Every model's overhead amounts side by side, so it is clear whether the
    # models actually disagree. Each is its own 3-hour cache entry, so switching
    # models never refetches. Plain floats: a numpy float (first load, from the
    # URL) and a Python float (after any widget change) would otherwise be two
    # cache entries for one place.
    _place = (round(float(lat), 2), round(float(lon), 2))
    _forecast_by_model, _compare_lines = {}, []
    for _key, _label in CLOUD_MODEL_KEYS.items():
        _mark = "▸ " if _key == cloud_model else ""
        try:
            _series = cached_cloud_series(*_place, _label)
            if _series is None:
                raise RuntimeError("fetch failed")
            _forecast_by_model[_key] = clouds.cover_at_time(_series, _when)
            _o = 100 * _forecast_by_model[_key][:, 0, 0]
            _compare_lines.append(f"{_mark}{_label}: low {_o[0]:.0f}% · mid {_o[1]:.0f}% · high {_o[2]:.0f}%")
        except ValueError:
            _compare_lines.append(f"{_mark}{_label}: no forecast for this time")
        except RuntimeError:
            _compare_lines.append(f"{_mark}{_label}: unavailable")
    with st.sidebar.expander("Adjust forecast amounts", expanded=False):
        cloud_adj = [
            st.slider("Adjust Low Clouds (points)", -100.0, 100.0, step=10.0, key="cloud_adj_low"),
            st.slider("Adjust Mid Clouds (points)", -100.0, 100.0, step=10.0, key="cloud_adj_mid"),
            st.slider("Adjust High Clouds (points)", -100.0, 100.0, step=10.0, key="cloud_adj_high"),
        ]
    if cloud_model in _forecast_by_model:
        cloud_cover_grid = clouds.adjust_cover(_forecast_by_model[cloud_model], cloud_adj)
    st.sidebar.caption(
        f"Overhead at {obs_time.strftime('%I:%M %p').lstrip('0')}:  \n" + "  \n".join(_compare_lines)
        + "  \nCloud shapes are illustrative and stay the same between models; the amount and where it "
          "sits (by direction) follow the selected model."
        + ("" if cloud_cover_grid is not None else "  \nNo forecast from the selected model: drawing a clear sky.")
    )
elif clouds_mode == "manual":
    cloud_man = [
        st.sidebar.slider("Low Clouds (%)", 0.0, 100.0, step=5.0, key="cloud_man_low"),
        st.sidebar.slider("Mid Clouds (%)", 0.0, 100.0, step=5.0, key="cloud_man_mid"),
        st.sidebar.slider("High Clouds (%)", 0.0, 100.0, step=5.0, key="cloud_man_high"),
    ]
    cloud_cover_grid = clouds.uniform_cover(*(v / 100.0 for v in cloud_man))

# --- SIDEBAR: 4. LABELS & OVERLAYS ---
st.sidebar.header("4. Labels & Overlays")

show_labels = st.sidebar.checkbox("Show Planet/Moon Labels", value=True)
show_major_star_labels = st.sidebar.checkbox("Show Major Star Labels", value=True)
show_minor_star_labels = st.sidebar.checkbox("Show Minor Star Labels", value=False)
show_constellations = st.sidebar.checkbox("Show Constellation Lines", value=True)
show_asterisms = st.sidebar.checkbox(
    "Show Asterisms", value=False,
    help="Familiar patterns that cross constellations or get lost inside a full figure: Big Dipper "
         "(with its pointers to Polaris), Summer and Winter Triangles, Winter Hexagon, Orion's Belt, "
         "and the Great Square.",
)
show_asterism_names = st.sidebar.checkbox("Show Asterism Names", value=False)
show_altitude_gridlines = st.sidebar.checkbox("Show Altitude Gridlines", value=False)
show_compass_lines = st.sidebar.checkbox("Show Compass Direction Lines", value=False)
show_safe_zones = False
if FORMATS[image_format]["safe_zone"]:
    show_safe_zones = st.sidebar.checkbox(
        "Show Social Safe Zones", value=False,
        help="Marks, on screen only, where Stories/Reels/feed interfaces cover the image so labels and the "
             "Moon can stay clear of them. The guide is not in the downloaded file.",
    )

# Persist the full view in the URL so a generated graphic is shareable/bookmarkable.
st.query_params["lat"] = f"{lat:.2f}"
st.query_params["lon"] = f"{lon:.2f}"
st.query_params["date"] = obs_date.isoformat()
st.query_params["time"] = obs_time.strftime("%H:%M")
st.query_params["tz"] = selected_tz
st.query_params["bearing"] = f"{bearing:.0f}"
st.query_params["fov"] = f"{fov_width:.0f}"
st.query_params["altmax"] = f"{alt_max:.0f}"
st.query_params["lock"] = "1" if lock_proportions else "0"
st.query_params["fmt"] = image_format
st.query_params["turbidity"] = f"{turbidity:.1f}"
st.query_params["brightness"] = f"{star_brightness:.1f}"
st.query_params["leaves"] = foliage
st.query_params["clouds"] = clouds_mode
if clouds_mode == "forecast":
    st.query_params["cmodel"] = st.session_state["cloud_model"]
    st.query_params["cadj"] = ",".join(f"{st.session_state[k]:.0f}" for k in ("cloud_adj_low", "cloud_adj_mid", "cloud_adj_high"))
elif clouds_mode == "manual":
    st.query_params["cman"] = ",".join(f"{st.session_state[k]:.0f}" for k in ("cloud_man_low", "cloud_man_mid", "cloud_man_high"))

with st.sidebar.expander("🔗 Shareable Link"):
    st.caption("This exact view (location, time, direction, and sky settings) is captured below. Append it to this app's base URL to share or bookmark it -- or just copy the address bar, which already reflects it.")
    _share_query = "&".join(f"{k}={v}" for k, v in st.query_params.items())
    st.code(f"?{_share_query}", language=None)


# Object labels sit above the horizon silhouette (zorder 100) so a body low
# behind the tree line is still identified, with a dark outline so the text
# reads against bright twilight and the silhouette alike.
LABEL_ZORDER = 110
# Gridlines and compass lines sit above the clouds (60) and everything under
# them, but below the horizon scene (100), which hides them like any distant object.
OVERLAY_LINE_ZORDER, OVERLAY_TEXT_ZORDER = 70, 71
LABEL_OUTLINE = [patheffects.withStroke(linewidth=2.5, foreground="#000000", alpha=0.55)]


def disk_vertical_radius(r_x_deg, fov_width, alt_max):
    """Altitude-axis radius that makes a disk with azimuth-axis radius
    `r_x_deg` look round on screen, given that the x and y axes have
    different degrees-per-inch (fov_width over the figure width vs.
    alt_max over its height)."""
    return r_x_deg * (FIG_WIDTH_IN / fov_width) / (FIG_HEIGHT_IN / alt_max)


def plot_point_sources(ax, az, alt, eff_mag, rgb, alpha, zorder):
    """Stars/planets: marker size from (extinction-adjusted) magnitude,
    per-point color and opacity, plus a soft two-layer glow on anything
    brighter than about magnitude 1 so the brightest objects read as
    brilliant rather than just larger dots."""
    az, alt, eff_mag, alpha = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (az, alt, eff_mag, alpha))
    rgb = np.atleast_2d(np.asarray(rgb, dtype=float))
    sizes = stars.marker_size(eff_mag)

    glow_strength = alpha * np.clip((1.0 - eff_mag) / 4.0, 0.0, 1.0)
    has_glow = glow_strength > 0
    if has_glow.any():
        # Many faint, widening layers so the glow's edge isn't visible as a ring.
        for size_mult, glow_alpha in ((2.0, 0.10), (3.5, 0.07), (5.5, 0.05), (8.0, 0.035), (11.0, 0.025)):
            ax.scatter(
                az[has_glow], alt[has_glow], s=sizes[has_glow] * size_mult,
                c=np.column_stack([rgb[has_glow], glow_alpha * glow_strength[has_glow]]),
                linewidths=0, zorder=zorder - 1,
            )
    ax.scatter(az, alt, s=sizes, c=np.column_stack([rgb, alpha]), linewidths=0, zorder=zorder)


def draw_disk_glow(ax, x, y, r_x, r_y, rgb, peak_alpha, extent_radii, zorder):
    """Soft glow around the Sun/Moon: a smooth Gaussian falloff from the
    disk's edge out to `extent_radii` disk radii, drawn as one RGBA image
    (stacked flat ellipses show visible rings)."""
    u = np.linspace(-extent_radii, extent_radii, 96)
    U, V = np.meshgrid(u, u)
    beyond_edge = np.maximum(np.hypot(U, V) - 1.0, 0.0)
    sigma = (extent_radii - 1.0) / 2.5
    glow = np.zeros(U.shape + (4,))
    glow[..., :3] = rgb
    glow[..., 3] = peak_alpha * np.exp(-(beyond_edge / sigma) ** 2)
    ax.imshow(glow, extent=[x - extent_radii * r_x, x + extent_radii * r_x,
                            y - extent_radii * r_y, y + extent_radii * r_y],
              origin="lower", aspect="auto", interpolation="bilinear", zorder=zorder)


# The real Sun is dimmer and warmer toward its edge (limb darkening; the
# visible-light coefficient is ~0.6, tempered here so the disk still reads
# as brilliant). The rim is kept gentle and the edge feathered: to the eye the
# Sun is a dazzling blob that melts into its glare, not a sharp-edged sticker.
SUN_LIMB_DARKENING = 0.22
SUN_LIMB_WARM_TINT = np.array([1.0, 0.84, 0.56])
SUN_EDGE_FEATHER = 0.07  # fraction of the radius over which the edge fades out


def draw_limb_darkened_disk(ax, x, y, r_x, r_y, rgb, zorder):
    u = np.linspace(-1.0, 1.0, 128)
    U, V = np.meshgrid(u, u)
    r = np.hypot(U, V)
    mu = np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))
    toward_limb = (1.0 - mu)[..., None]
    intensity = 1.0 - SUN_LIMB_DARKENING * toward_limb
    tint = 1.0 - toward_limb * (1.0 - SUN_LIMB_WARM_TINT)
    disk = np.zeros(U.shape + (4,))
    disk[..., :3] = np.clip(np.asarray(rgb) * intensity * tint, 0.0, 1.0)
    disk[..., 3] = np.clip((1.0 - r) / SUN_EDGE_FEATHER, 0.0, 1.0)  # feathered edge
    ax.imshow(disk, extent=[x - r_x, x + r_x, y - r_y, y + r_y],
              origin="lower", aspect="auto", interpolation="bilinear", zorder=zorder)


# --- GRAPHIC GENERATION LOGIC ---
if st.button("Generate Sky Graphic", type="primary"):
    with st.spinner("Computing high-fidelity directional sky model and celestial structures..."):

        dt_local = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
        dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
        t = ts.from_datetime(dt_utc)
        observer_loc = earth + wgs84.latlon(lat, lon)

        sun_astrometric = observer_loc.at(t).observe(sun)
        sun_apparent = sun_astrometric.apparent()
        sun_alt, sun_az, sun_distance = sun_apparent.altaz(temperature_C=STANDARD_TEMPERATURE_C, pressure_mbar=STANDARD_PRESSURE_MBAR)
        sun_deg = sun_alt.degrees
        sun_az_deg = sun_az.degrees

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=100, facecolor='none')
        ax.set_position([0, 0, 1, 1])  # full-bleed: no default subplot margins around the sky
        ax.set_xlim(az_min, az_max)
        ax.set_ylim(0, alt_max)

        # ======================================================================
        # SECTION 3: SKY BACKGROUND
        # One physically-based model (astro/atmosphere.py) for day, sunset,
        # twilight and night: sunlight and moonlight scattered by air, haze
        # and ozone around a spherical Earth, plus natural airglow and
        # light-pollution skyglow. Sunset color, Earth's shadow and the Belt
        # of Venus come out of the geometry rather than being painted on.
        # ======================================================================
        moon_sky_kwargs = {}
        try:
            moon_for_sky = observer_loc.at(t).observe(eph['moon']).apparent()
            ms_alt, ms_az, ms_distance = moon_for_sky.altaz(temperature_C=STANDARD_TEMPERATURE_C, pressure_mbar=STANDARD_PRESSURE_MBAR)
            # The Moon's phase angle is ~180deg minus its elongation from the Sun.
            moon_sky_kwargs = dict(
                moon_alt_deg=float(ms_alt.degrees), moon_az_deg=float(ms_az.degrees),
                moon_phase_angle_deg=180.0 - float(moon_for_sky.separation_from(sun_apparent).degrees),
                moon_distance_km=float(ms_distance.km),
            )
        except Exception:
            logger.exception("Moon position for sky brightness failed; rendering without moonlight")

        sky_display, sky_luminance, sky_zenith_dark, sky_rgb_linear, _sky_zenith_total = cached_sky(
            az_min, az_max, alt_max, sun_deg, sun_az_deg, turbidity, star_brightness, **moon_sky_kwargs
        )
        # The sky is computed on a coarse grid; bilinear upscaling keeps the
        # gradient smooth (the default would nearest-neighbor it into stripes).
        ax.imshow(sky_display, extent=[az_min, az_max, 0, alt_max], origin="lower", aspect="auto",
                  interpolation="bilinear", zorder=0)

        # Clouds sit in front of the stars, planets, Sun and Moon (they fade or
        # vanish behind cloud) but under the horizon scene and the labels, which
        # stay readable.
        if cloud_cover_grid is not None and cloud_cover_grid.max() > 0:
            try:
                cloud_seed = int(abs(round(lat, 2) * 1000) + abs(round(lon, 2) * 1000)
                                 + obs_date.toordinal() * 24 + obs_time.hour) % (2 ** 31)
                moon_light = {}
                if moon_sky_kwargs:
                    moon_light = dict(
                        moon_alt=moon_sky_kwargs["moon_alt_deg"], moon_az=moon_sky_kwargs["moon_az_deg"],
                        moon_illuminance=atmosphere.moon_illuminance_lux(
                            moon_sky_kwargs["moon_phase_angle_deg"], moon_sky_kwargs["moon_distance_km"]),
                    )
                cloud_rgba = cached_clouds(cloud_cover_grid, az_min, az_max, alt_max, sun_deg, sun_az_deg,
                                           turbidity, star_brightness, sky_rgb_linear, cloud_seed,
                                           *cloud_grid_size(image_format),
                                           **moon_light, **moon_sky_kwargs)
                ax.imshow(cloud_rgba, extent=[az_min, az_max, 0, alt_max], origin="lower", aspect="auto",
                          interpolation="bilinear", zorder=60)
            except Exception:
                logger.exception("Cloud rendering failed; drawing a clear sky")

        # Reference-overlay styling (altitude gridlines, compass lines):
        # needs to stay legible against both a bright daytime sky and a
        # near-black night sky, so rather than guess a single muted hue
        # per regime, pick a contrasting fill + an opposite-toned outline
        # stroke -- dark-on-light for day, light-on-dark for night/twilight.
        if sun_deg > 0:
            grid_line_color, grid_text_color, grid_stroke_color = "#1e293b", "#1e293b", "#ffffff"
        else:
            grid_line_color, grid_text_color, grid_stroke_color = "#e2e8f0", "#e2e8f0", "#0a0f18"
        grid_outline = [patheffects.withStroke(linewidth=1.6, foreground=grid_stroke_color, alpha=0.9)]

        # Faintest magnitude visible at each point in the sky. Twilight onset
        # stays anchored to the empirical sun-altitude table (the eye does
        # better in bright twilight than the standard sky-brightness formula
        # predicts); on top of that, anything making a spot brighter than a
        # moonless, unpolluted sky -- the Moon and its glow, light pollution,
        # the bright twilight arch -- lowers the limit there by the same
        # amount the sky-brightness formula says it should.
        twilight_limit = stars.twilight_limiting_magnitude(sun_deg)
        dark_reference_limit = atmosphere.naked_eye_limit(sky_zenith_dark)

        def limiting_magnitude_at(az, alt):
            ix = np.round((np.asarray(az, dtype=float) - az_min) / (az_max - az_min) * (SKY_GRID_W - 1))
            iy = np.round(np.asarray(alt, dtype=float) / alt_max * (SKY_GRID_H - 1))
            ix = np.clip(ix, 0, SKY_GRID_W - 1).astype(int)
            iy = np.clip(iy, 0, SKY_GRID_H - 1).astype(int)
            return twilight_limit - (dark_reference_limit - atmosphere.naked_eye_limit(sky_luminance[iy, ix]))

        # Horizon geometry, computed up front (drawn in section 6) so labels
        # can skip objects hidden behind it. Two layers, both rising from
        # the baseline: distant terrain from real elevation data (when the
        # lookup succeeds), and the nearby tree/house silhouette in front of
        # it. Their union is what an observer sees -- foreground trees hide
        # low distant terrain, and mountains taller than the trees rise
        # above them. The foreground scene (city, suburb or countryside)
        # follows the Star Visibility Limit.
        x_silhouette_space = np.linspace(az_min, az_max, 400)
        try:
            sample_az, horizon_deg_samples = cached_horizon_profile(lat, lon, az_min, az_max)
            terrain_top = np.clip(np.interp(x_silhouette_space, sample_az, horizon_deg_samples), 0.3, alt_max)
        except Exception:
            logger.exception("Terrain horizon lookup failed; drawing the foreground silhouette only")
            terrain_top = None

        treeline_seed = int(abs(lat * 10007 + lon * 7919 + bearing * 104729)) % (2 ** 32)
        horizon_scene = treeline.scene_for_visibility(star_brightness)
        horizon_strips, horizon_is_bare = get_horizon_strips(horizon_scene, horizon_has_leaves)
        if not horizon_has_leaves and not horizon_is_bare and horizon_scene != "urban":
            st.info(f"Winter (bare-tree) art isn't available for the {horizon_scene} scene yet; "
                    "showing it with leaves.")
        treeline_rgba, treeline_extent = treeline.tiled_treeline(
            horizon_strips, az_min, az_max, treeline_seed, height_deg=treeline.scene_height_deg(horizon_scene),
        )

        _fg_az, _fg_top = treeline.silhouette_top_profile(treeline_rgba, treeline_extent)
        horizon_top_profile = np.interp(x_silhouette_space, _fg_az, _fg_top)
        if terrain_top is not None:
            horizon_top_profile = np.maximum(horizon_top_profile, terrain_top)

        def horizon_top_at(az):
            """Altitude the horizon silhouette reaches at azimuth `az`."""
            return np.interp(az, x_silhouette_space, horizon_top_profile)

        # --- SUN DISK ---
        # Same exaggeration as the Moon so their relative size stays true;
        # reddened and refraction-flattened near the horizon, and clipped
        # by the horizon silhouette (zorder 100) as it sets.
        sun_true_radius = moon.angular_radius_deg(sun_distance.km, moon.SUN_RADIUS_KM)
        sun_r_x = sun_true_radius * moon.DISK_DISPLAY_SCALE
        sun_r_y = disk_vertical_radius(sun_r_x, fov_width, alt_max) * moon.refraction_squash(sun_deg, sun_true_radius)
        sun_plot_az = float(normalize_az(sun_az_deg, az_min, az_max))
        if az_min <= sun_plot_az <= az_max and -sun_r_y <= sun_deg <= alt_max + sun_r_y:
            # The Sun's light is physically reddened at any altitude, but it's
            # dazzling (reads as white) until it's low and dim enough to look
            # at, so ease from warm white up high to the full tint at the horizon.
            reddening = np.clip(1.0 - sun_deg / 15.0, 0.25, 1.0)
            sun_tint = np.array([1.0, 0.98, 0.92]) * extinction.color_tint(max(sun_deg, 0.0), turbidity)
            sun_rgb = 1.0 - reddening * (1.0 - sun_tint)
            draw_disk_glow(ax, sun_plot_az, sun_deg, sun_r_x, sun_r_y, sun_rgb,
                           peak_alpha=0.45, extent_radii=8.0, zorder=44)
            draw_limb_darkened_disk(ax, sun_plot_az, sun_deg, sun_r_x, sun_r_y, sun_rgb, zorder=46)

        # 4. PLOT PLANETS & DYNAMIC MOON ENGINE
        bodies = {
            'mercury': (eph['mercury'], 'Mercury'),
            'venus': (eph['venus'], 'Venus'),
            'mars': (eph['mars'], 'Mars'),
            'jupiter': (eph['jupiter_barycenter'], 'Jupiter'),
            'saturn': (eph['saturn_barycenter'], 'Saturn'),
        }

        for name, (body, label) in bodies.items():
            try:
                astrometric = observer_loc.at(t).observe(body)
                alt, az, _ = astrometric.apparent().altaz(temperature_C=STANDARD_TEMPERATURE_C, pressure_mbar=STANDARD_PRESSURE_MBAR)

                body_az, body_alt = az.degrees, alt.degrees

                body_az = normalize_az(body_az, az_min, az_max)

                if az_min <= body_az <= az_max and 0 <= body_alt <= alt_max:
                    mag = float(planetary_magnitude(astrometric))
                    if np.isnan(mag):  # outside the magnitude model's phase-angle range
                        mag = 0.0
                    eff_mag = mag + extinction.magnitude_loss(body_alt, turbidity)
                    alpha = float(stars.visibility_alpha(eff_mag, limiting_magnitude_at(body_az, body_alt)))
                    if alpha <= 0.0:
                        continue
                    body_rgb = np.array(stars.PLANET_COLORS[label]) * extinction.color_tint(body_alt, turbidity)
                    # Below the Moon (zorder 49-50) so a lunar occultation hides the planet.
                    plot_point_sources(ax, body_az, body_alt, eff_mag, body_rgb, alpha, zorder=48)
                    if show_labels and body_alt > horizon_top_at(body_az):
                        ax.text(body_az + 0.5, body_alt + 0.5, label, color="#ffffff", fontsize=10, weight='bold',
                                path_effects=LABEL_OUTLINE, zorder=LABEL_ZORDER)
            except Exception:
                logger.exception("Failed to compute/plot position for %s", label)
                continue

        # --- DYNAMIC ROTATIONAL MOON PHASE VECTOR PATH ENGINE ---
        try:
            moon_body = eph['moon']
            moon_astrometric = observer_loc.at(t).observe(moon_body)
            moon_apparent = moon_astrometric.apparent()
            m_alt, m_az, m_distance = moon_apparent.altaz(temperature_C=STANDARD_TEMPERATURE_C, pressure_mbar=STANDARD_PRESSURE_MBAR)

            moon_az, moon_alt = m_az.degrees, m_alt.degrees

            moon_az = normalize_az(moon_az, az_min, az_max)

            moon_true_radius = moon.angular_radius_deg(m_distance.km)
            r_x = moon_true_radius * moon.DISK_DISPLAY_SCALE  # visually exaggerated for broadcast legibility
            r_y = disk_vertical_radius(r_x, fov_width, alt_max) * moon.refraction_squash(moon_alt, moon_true_radius)

            if az_min <= moon_az <= az_max and -r_y <= moon_alt <= alt_max:
                m_pos = observer_loc.at(t).observe(moon_body).position.au
                s_pos = observer_loc.at(t).observe(sun).position.au

                m_dot_s = np.dot(m_pos, s_pos) / (np.linalg.norm(m_pos) * np.linalg.norm(s_pos))
                elongation = np.arccos(np.clip(m_dot_s, -1.0, 1.0))
                # Phase angle ~= 180deg - elongation for the Moon, so the lit
                # fraction is (1 - cos(elongation)) / 2: 0 at new, 1 at full.
                illuminated_fraction = 0.5 * (1.0 - np.cos(elongation))

                pabl_rad = moon.bright_limb_plot_angle_rad(sun_az_deg, sun_deg, moon_az, moon_alt)

                phi = np.linspace(-np.pi/2, np.pi/2, 30)

                x_outer_unit, y_outer_unit = np.cos(phi), np.sin(phi)

                # Terminator half-width as a fraction of the radius: +1 puts it
                # on the lit limb (new Moon, nothing lit), -1 on the far limb (full).
                phase_modifier = (0.5 - illuminated_fraction) * 2.0
                x_inner_unit, y_inner_unit = x_outer_unit * phase_modifier, y_outer_unit

                cos_p, sin_p = np.cos(pabl_rad), np.sin(pabl_rad)

                # VECTORIZED MOON TERMINATOR MATH
                x_out_rot = x_outer_unit * cos_p - y_outer_unit * sin_p
                y_out_rot = x_outer_unit * sin_p + y_outer_unit * cos_p

                x_in_rot = (x_inner_unit * cos_p - y_inner_unit * sin_p)[::-1]
                y_in_rot = (x_inner_unit * sin_p + y_inner_unit * cos_p)[::-1]

                x_verts = np.concatenate([x_out_rot, x_in_rot]) * r_x + moon_az
                y_verts = np.concatenate([y_out_rot, y_in_rot]) * r_y + moon_alt

                verts = np.column_stack((x_verts, y_verts))
                verts = np.vstack((verts, verts[0])) # Close polygon

                codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
                moon_path = Path(verts, codes)

                # Rising/setting Moon reddens toward orange like the Sun does.
                moon_rgb = np.array([0.97, 0.96, 0.92]) * extinction.color_tint(max(moon_alt, 0.0), turbidity)

                # Soft halo from atmospheric scattering, brighter the fuller the Moon.
                draw_disk_glow(ax, moon_az, moon_alt, r_x, r_y, moon_rgb,
                               peak_alpha=0.30 * illuminated_fraction, extent_radii=4.0, zorder=47)

                moon_patch = patches.PathPatch(moon_path, facecolor=moon_rgb, edgecolor='none', zorder=50)
                ax.add_patch(moon_patch)

                if illuminated_fraction < 0.90:
                    dark_disk = patches.Ellipse((moon_az, moon_alt), width=r_x*2, height=r_y*2, facecolor=moon_rgb, alpha=0.08, edgecolor='none', zorder=49)
                    ax.add_patch(dark_disk)

                # Labeled once at least its upper half clears the horizon silhouette.
                if show_labels and moon_alt > horizon_top_at(moon_az):
                    ax.text(moon_az + r_x + 0.4, moon_alt + 0.6, "Moon", color="#ffffff", fontsize=11, weight='bold',
                            path_effects=LABEL_OUTLINE, zorder=LABEL_ZORDER)
        except Exception:
            logger.exception("Failed to compute/plot Moon position")

        # 5. DYNAMIC HIPPARCOS STAR FIELD
        # Stars fade in progressively through twilight (brightest first)
        # rather than all appearing at once, capped by the Star Visibility
        # Limit slider. Catalog limits are zenith values, so the cheap
        # magnitude pre-filter below is safe: extinction only ever dims.
        # Cheap pre-filter; the per-star limit below can only be a little
        # above the zenith reference (spots darker than the zenith).
        drawn_hips = set()
        visible_stars = stars_df[stars_df['magnitude'] <= min(star_brightness, twilight_limit + 1.0)]
        if len(visible_stars):
            star_obj = Star.from_dataframe(visible_stars)

            # Vectorized altitude/azimuth calculations for the entire visible catalog
            star_astrometric = observer_loc.at(t).observe(star_obj)
            s_alt, s_az, _ = star_astrometric.apparent().altaz(temperature_C=STANDARD_TEMPERATURE_C, pressure_mbar=STANDARD_PRESSURE_MBAR)

            star_az = normalize_az(s_az.degrees, az_min, az_max)
            star_alt = s_alt.degrees

            # Atmospheric extinction dims stars toward the horizon (so they
            # thin out there, as in reality) and reddens what's left.
            star_eff_mag = visible_stars['magnitude'].values + extinction.magnitude_loss(star_alt, turbidity)
            star_limit = np.minimum(star_brightness, limiting_magnitude_at(star_az, star_alt))
            star_alpha = stars.visibility_alpha(star_eff_mag, star_limit)

            viewport_mask = (
                (star_az >= az_min) & (star_az <= az_max) & (star_alt >= 0) & (star_alt <= alt_max)
                & (star_alpha > 0)
            )

            plot_az = star_az[viewport_mask]
            plot_alt = star_alt[viewport_mask]
            plot_hips = visible_stars.index.values[viewport_mask]
            drawn_hips.update(plot_hips.tolist())
            plot_rgb = (stars.bv_to_rgb(visible_stars['bv'].values[viewport_mask])
                        * extinction.color_tint(plot_alt, turbidity))

            plot_point_sources(ax, plot_az, plot_alt, star_eff_mag[viewport_mask], plot_rgb,
                               star_alpha[viewport_mask], zorder=20)

            # Dictionary of major anchor stars (Hipparcos ID -> Common Name)
            major_stars = {
                32349: "Sirius", 24608: "Capella", 69673: "Arcturus", 91262: "Vega",
                24436: "Rigel", 37279: "Procyon", 27989: "Betelgeuse", 97649: "Altair",
                21421: "Aldebaran", 65474: "Spica", 80763: "Antares", 37826: "Pollux",
                102098: "Deneb", 49669: "Regulus", 36850: "Castor", 11767: "Polaris"
            }

            # Only label stars that aren't hidden behind the horizon silhouette.
            unobstructed = plot_alt > horizon_top_at(plot_az)
            label_az, label_alt, label_hips = plot_az[unobstructed], plot_alt[unobstructed], plot_hips[unobstructed]

            if show_major_star_labels:
                for az_val, alt_val, hip_id in zip(label_az, label_alt, label_hips):
                    if hip_id in major_stars:
                        ax.text(az_val + 0.4, alt_val + 0.4, major_stars[hip_id], color="#ffffff", fontsize=9, alpha=0.5,
                                path_effects=LABEL_OUTLINE, zorder=LABEL_ZORDER)

            if show_minor_star_labels:
                for az_val, alt_val, hip_id in zip(label_az, label_alt, label_hips):
                    if hip_id not in major_stars:
                        ax.text(az_val + 0.4, alt_val + 0.4, f"HIP {hip_id}", color="#ffffff", fontsize=7, alpha=0.35,
                                path_effects=LABEL_OUTLINE, zorder=LABEL_ZORDER)

        # --- CONSTELLATION STICK FIGURES AND ASTERISMS ---
        # Reference overlays, independent of the star limit above. The stars
        # that form each figure are always drawn (faintly) with it, so the lines
        # never float without anything to point at -- even when twilight or the
        # Star Visibility Limit would hide those stars. Lines and guide stars
        # fade in over the first few degrees of deepening twilight. Asterisms
        # are drawn warmer and heavier than constellations so both read together.
        if (show_constellations or show_asterisms) and sun_deg <= -6:
            try:
                guide_fade = 0.3 + 0.7 * float(np.clip((-6.0 - sun_deg) / 4.0, 0.0, 1.0))
                wanted_hips = set()
                if show_constellations:
                    wanted_hips |= {hip for pair in CONSTELLATION_SEGMENTS for hip in pair}
                if show_asterisms:
                    wanted_hips |= asterisms.all_hips()
                seg_stars = stars_df.loc[stars_df.index.intersection(sorted(wanted_hips))]
                seg_star_obj = Star.from_dataframe(seg_stars)
                seg_astrometric = observer_loc.at(t).observe(seg_star_obj)
                seg_alt, seg_az, _ = seg_astrometric.apparent().altaz(temperature_C=STANDARD_TEMPERATURE_C, pressure_mbar=STANDARD_PRESSURE_MBAR)
                seg_az_deg = normalize_az(seg_az.degrees, az_min, az_max)
                seg_alt_deg = seg_alt.degrees
                pos_by_hip = dict(zip(seg_stars.index.values, zip(seg_az_deg, seg_alt_deg)))

                def seg_in_view(pos):
                    return az_min <= pos[0] <= az_max and 0 <= pos[1] <= alt_max

                def visible_pair(hip_a, hip_b):
                    """Both end positions when the line is fully in view, else None."""
                    pa, pb = pos_by_hip.get(hip_a), pos_by_hip.get(hip_b)
                    return (pa, pb) if pa is not None and pb is not None and seg_in_view(pa) and seg_in_view(pb) else None

                anchored = set()
                if show_constellations:
                    for hip_a, hip_b in CONSTELLATION_SEGMENTS:
                        pair = visible_pair(hip_a, hip_b)
                        if pair is None:
                            continue
                        (az_a, alt_a), (az_b, alt_b) = pair
                        ax.plot([az_a, az_b], [alt_a, alt_b], color="#7dd3fc", alpha=0.35 * guide_fade,
                                linewidth=0.8, zorder=15)
                        anchored.update((hip_a, hip_b))

                if show_asterisms:
                    for asterism in asterisms.ASTERISMS:
                        drawn_points = []  # solid lines only: a dashed pointer must not drag the name away
                        for hip_a, hip_b, dashed in asterisms.lines(asterism):
                            pair = visible_pair(hip_a, hip_b)
                            if pair is None:
                                continue
                            (az_a, alt_a), (az_b, alt_b) = pair
                            ax.plot([az_a, az_b], [alt_a, alt_b], color="#fcd34d", alpha=0.7 * guide_fade,
                                    linewidth=1.3, linestyle=(0, (4, 3)) if dashed else "-", zorder=16)
                            anchored.update((hip_a, hip_b))
                            if not dashed:
                                drawn_points += [(az_a, alt_a), (az_b, alt_b)]
                        if show_asterism_names and drawn_points:
                            name_az = float(np.mean([pt[0] for pt in drawn_points]))
                            name_alt = max(pt[1] for pt in drawn_points) + 1.5
                            if name_alt <= alt_max and name_alt > horizon_top_at(name_az):
                                ax.text(name_az, name_alt, asterism["name"], color="#fde68a", fontsize=9,
                                        ha="center", va="bottom", alpha=0.85 * guide_fade,
                                        path_effects=LABEL_OUTLINE, zorder=LABEL_ZORDER)

                # Guide stars at the figures' vertices, except those the regular
                # star pass above already drew at their true brightness.
                guide = [h for h in sorted(anchored) if h not in drawn_hips]
                if guide:
                    g_stars = seg_stars.loc[guide]
                    g_az = np.array([pos_by_hip[h][0] for h in guide])
                    g_alt = np.array([pos_by_hip[h][1] for h in guide])
                    # Sized as at worst a magnitude-3 star so even the faintest vertex
                    # is a readable anchor on a broadcast-size frame.
                    g_mag = np.minimum(g_stars['magnitude'].values + extinction.magnitude_loss(g_alt, turbidity), 3.0)
                    g_rgb = stars.bv_to_rgb(g_stars['bv'].values) * extinction.color_tint(g_alt, turbidity)
                    plot_point_sources(ax, g_az, g_alt, g_mag, g_rgb,
                                       np.full(len(guide), 0.7 * guide_fade), zorder=19)
            except Exception:
                logger.exception("Failed to draw constellation lines/asterisms")

        # 6. HORIZON SILHOUETTE (geometry computed before section 4)
        if terrain_top is not None:
            ax.fill_between(x_silhouette_space, -5, terrain_top, color="#060c14", zorder=100)
        ax.imshow(treeline_rgba, extent=treeline_extent, aspect="auto", zorder=100)

        # Reference overlays are opt-in and drawn manually (axhline/axvline
        # + text), matching how every other label in this chart is drawn --
        # native tick labels are never used, since both axes stay hidden.
        if show_altitude_gridlines:
            for alt_tick in np.arange(10, alt_max, 10):
                ax.axhline(alt_tick, color=grid_line_color, alpha=0.75, linestyle='--',
                           linewidth=1.1, path_effects=grid_outline, zorder=OVERLAY_LINE_ZORDER)
                ax.text(az_min + 0.5, alt_tick + 0.3, f"{int(alt_tick)}°", color=grid_text_color,
                        fontsize=9, alpha=0.95, path_effects=grid_outline, zorder=OVERLAY_TEXT_ZORDER)

        if show_compass_lines:
            for compass_az, compass_label in compass_ticks(az_min, az_max):
                ax.axvline(compass_az, color=grid_line_color, alpha=0.75, linestyle='--',
                           linewidth=1.1, path_effects=grid_outline, zorder=OVERLAY_LINE_ZORDER)
                ax.text(compass_az, alt_max - 1.5, compass_label, color=grid_text_color, ha='center',
                        fontsize=9, alpha=0.95, path_effects=grid_outline, zorder=OVERLAY_TEXT_ZORDER)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        # The download is the exact PNG of the chosen format. The axes fill the
        # figure, so no "tight" cropping (that, and st.pyplot's default 0.1"
        # padding, added a transparent margin that showed up as a white border).
        dpi = export_dpi(image_format)
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=dpi, facecolor="none", edgecolor="none")
        img_buf.seek(0)

        # Safe-zone guides are for the screen only: drawn after the download
        # image is saved, so they never appear in the file.
        preview_bytes = img_buf.getvalue()
        zone = FORMATS[image_format]["safe_zone"]
        if show_safe_zones and zone:
            def guide_band(x0, y0, width, height):
                ax.add_patch(patches.Rectangle((x0, y0), width, height, transform=ax.transAxes,
                                               facecolor="#ff3b3b", edgecolor="none", alpha=0.20, zorder=500))
            guide_band(0.0, 1.0 - zone["top"], 1.0, zone["top"])
            guide_band(0.0, 0.0, 1.0, zone["bottom"])
            if zone["side"] > 0:
                guide_band(0.0, 0.0, zone["side"], 1.0)
                guide_band(1.0 - zone["side"], 0.0, zone["side"], 1.0)
            preview = io.BytesIO()
            fig.savefig(preview, format="png", dpi=dpi, facecolor="none", edgecolor="none")
            preview_bytes = preview.getvalue()
        st.download_button(label="💾 Download High-Res PNG for Editing / On-Air", data=img_buf,
                           file_name=f"custom_sky_{bearing:.0f}deg" + ("" if image_format == DEFAULT_FORMAT else f"_{image_format}") + ".png",
                           mime="image/png")
        # Fit the preview to the window: scale to the smaller of the window width
        # and height (tall 9:16 / 4:5 images no longer need scrolling), centered.
        st.markdown(
            "<style>"
            "[data-testid='stFullScreenFrame'] {display: flex; justify-content: center;}"
            "[data-testid='stImage'] img {width: auto !important; max-width: 100%; "
            "max-height: calc(100vh - 140px); object-fit: contain; scroll-margin-top: 70px;}"
            "</style>",
            unsafe_allow_html=True,
        )
        try:
            st.image(preview_bytes, width="stretch")
        except Exception:  # older Streamlit without width="stretch"
            st.image(preview_bytes, use_container_width=True)
        # Bring the finished image to the top of the window.
        components.html(
            "<script>let n = 0; const t = setInterval(() => {"
            "const i = window.parent.document.querySelector(\"[data-testid='stImage'] img\");"
            "if (i && i.complete) {i.scrollIntoView({block: 'start'});}"
            "if (++n > 15) clearInterval(t);}, 150);</script>", height=0)
        if show_safe_zones and zone:
            st.caption("Red bands mark where social apps cover the image with their own interface. "
                       "They are a guide only and are not in the download.")
        plt.close(fig)

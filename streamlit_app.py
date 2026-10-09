import logging
import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")  # Safe headless execution for cloud servers
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import io
import numpy as np

# Core astronomical math engine
from skyfield.api import wgs84, Star, Loader
from skyfield.data import hipparcos
from skyfield.magnitudelib import planetary_magnitude

from astro import atmosphere, constellations, extinction, horizon, location, moon, stars, treeline

logger = logging.getLogger(__name__)

# Standard-atmosphere constants used to enable refraction for all altaz()
# calls below. This app has no live local weather input, so these are a
# reasonable fixed approximation -- still meaningfully more accurate near
# the horizon (this app's whole 0-40+deg window) than ignoring refraction
# entirely, which is what happened before.
STANDARD_TEMPERATURE_C = 10.0
STANDARD_PRESSURE_MBAR = 1010.0

# The graphic is drawn 16:9. A degree takes the same space horizontally and
# vertically (no stretching of trees, houses or the Moon) when the altitude
# shown is the field-of-view width times height/width.
FIG_WIDTH_IN, FIG_HEIGHT_IN = 12.0, 6.75
ALT_MAX_RANGE = (20.0, 90.0)
ALT_MAX_STEP = 5.0


def natural_altitude_for_fov(fov_deg):
    """Max altitude that gives equal-angle proportions for `fov_deg`,
    rounded to the altitude slider's step and limited to its range."""
    ideal = fov_deg * FIG_HEIGHT_IN / FIG_WIDTH_IN
    stepped = np.floor(ideal / ALT_MAX_STEP + 0.5) * ALT_MAX_STEP
    return float(np.clip(stepped, *ALT_MAX_RANGE))

# Set page layout to wide for a clean dashboard feel
st.set_page_config(layout="wide", page_title="Custom Sky Graphic Generator")

st.title("🌌 Broadcast Sky Graphic Generator")
st.write("A lightweight, reliable engine rendering clean astronomical plates with native horizon silhouettes.")

# ======================================================================
# CACHED DATA ENGINE
# Loads the heavy ephemeris and star catalog files into memory exactly once,
# preventing timeouts and massive latency delays on cloud deployments.
# ======================================================================
@st.cache_resource
def get_astronomy_data():
    loader = Loader('.')
    ts = loader.timescale()
    eph = loader('de421.bsp')

    # Forces Streamlit Cloud to download the catalog natively if missing
    with loader.open(hipparcos.URL) as f:
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
def get_horizon_strips(scene):
    return treeline.load_scene_strips(scene)


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
    return display, sky["luminance"], sky["zenith_dark"]


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

tz_options = ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]
url_tz = st.query_params.get("tz", tz_options[0])
selected_tz = st.sidebar.selectbox("Time Zone", tz_options, index=tz_options.index(url_tz) if url_tz in tz_options else 0)

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
try:
    url_fov = np.clip(float(st.query_params.get("fov", 80.0)), 30.0, 180.0)
except ValueError:
    url_fov = 80.0
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

st.sidebar.selectbox(
    "Direction Preset", list(DIRECTION_PRESETS.keys()),
    key="direction_preset", on_change=_apply_direction_preset,
)
bearing = st.sidebar.slider(
    "Compass Bearing (0=N, 90=E, 180=S, 270=W)", 0.0, 359.0, step=1.0,
    key="bearing", on_change=_clear_preset_if_bearing_diverged,
)
fov_width = st.sidebar.slider("Field of View Width (deg)", 30.0, 180.0, step=5.0, key="fov_width")
lock_proportions = st.sidebar.checkbox(
    "Lock altitude to natural proportions", key="lock_proportions",
    help="Keeps the altitude shown matched to the width (at the graphic's 16:9 shape) so a "
         "degree looks the same size sideways and upward -- trees, houses, and the Moon aren't "
         "stretched. Untick to set the altitude yourself.",
)
if lock_proportions:
    # Set before the slider below is created (Streamlit only allows that order).
    st.session_state["alt_max"] = natural_altitude_for_fov(fov_width)
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

# --- SIDEBAR: 4. LABELS & OVERLAYS ---
st.sidebar.header("4. Labels & Overlays")

show_labels = st.sidebar.checkbox("Show Planet/Moon Labels", value=True)
show_major_star_labels = st.sidebar.checkbox("Show Major Star Labels", value=True)
show_minor_star_labels = st.sidebar.checkbox("Show Minor Star Labels", value=False)
show_constellations = st.sidebar.checkbox("Show Constellation Lines", value=True)
show_altitude_gridlines = st.sidebar.checkbox("Show Altitude Gridlines", value=False)
show_compass_lines = st.sidebar.checkbox("Show Compass Direction Lines", value=False)

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
st.query_params["turbidity"] = f"{turbidity:.1f}"
st.query_params["brightness"] = f"{star_brightness:.1f}"

with st.sidebar.expander("🔗 Shareable Link"):
    st.caption("This exact view (location, time, direction, and sky settings) is captured below. Append it to this app's base URL to share or bookmark it -- or just copy the address bar, which already reflects it.")
    _share_query = "&".join(f"{k}={v}" for k, v in st.query_params.items())
    st.code(f"?{_share_query}", language=None)


# Object labels sit above the horizon silhouette (zorder 100) so a body low
# behind the tree line is still identified, with a dark outline so the text
# reads against bright twilight and the silhouette alike.
LABEL_ZORDER = 110
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

        sky_display, sky_luminance, sky_zenith_dark = cached_sky(
            az_min, az_max, alt_max, sun_deg, sun_az_deg, turbidity, star_brightness, **moon_sky_kwargs
        )
        ax.imshow(sky_display, extent=[az_min, az_max, 0, alt_max], origin="lower", aspect="auto", zorder=0)

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
        treeline_rgba, treeline_extent = treeline.tiled_treeline(
            get_horizon_strips(treeline.scene_for_visibility(star_brightness)),
            az_min, az_max, treeline_seed,
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

        # --- CONSTELLATION STICK FIGURES ---
        # Drawn from the end of civil twilight, as before -- independent of
        # the star limit above, since the figures are a reference overlay.
        if show_constellations and sun_deg <= -6:
            try:
                seg_hip_ids = sorted({hip for pair in CONSTELLATION_SEGMENTS for hip in pair})
                seg_stars = stars_df.loc[stars_df.index.intersection(seg_hip_ids)]
                seg_star_obj = Star.from_dataframe(seg_stars)
                seg_astrometric = observer_loc.at(t).observe(seg_star_obj)
                seg_alt, seg_az, _ = seg_astrometric.apparent().altaz(temperature_C=STANDARD_TEMPERATURE_C, pressure_mbar=STANDARD_PRESSURE_MBAR)
                seg_az_deg = normalize_az(seg_az.degrees, az_min, az_max)
                seg_alt_deg = seg_alt.degrees
                pos_by_hip = dict(zip(seg_stars.index.values, zip(seg_az_deg, seg_alt_deg)))

                for hip_a, hip_b in CONSTELLATION_SEGMENTS:
                    pa, pb = pos_by_hip.get(hip_a), pos_by_hip.get(hip_b)
                    if pa is None or pb is None:
                        continue
                    if not (az_min <= pa[0] <= az_max and 0 <= pa[1] <= alt_max):
                        continue
                    if not (az_min <= pb[0] <= az_max and 0 <= pb[1] <= alt_max):
                        continue
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#7dd3fc", alpha=0.35, linewidth=0.8, zorder=15)
            except Exception:
                logger.exception("Failed to draw constellation lines")

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
                           linewidth=1.1, path_effects=grid_outline, zorder=2)
                ax.text(az_min + 0.5, alt_tick + 0.3, f"{int(alt_tick)}°", color=grid_text_color,
                        fontsize=9, alpha=0.95, path_effects=grid_outline, zorder=3)

        if show_compass_lines:
            for compass_az, compass_label in compass_ticks(az_min, az_max):
                ax.axvline(compass_az, color=grid_line_color, alpha=0.75, linestyle='--',
                           linewidth=1.1, path_effects=grid_outline, zorder=2)
                ax.text(compass_az, alt_max - 1.5, compass_label, color=grid_text_color, ha='center',
                        fontsize=9, alpha=0.95, path_effects=grid_outline, zorder=3)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=150, facecolor="none", edgecolor="none", bbox_inches="tight", pad_inches=0.0)
        img_buf.seek(0)

        st.download_button(label="💾 Download High-Res PNG for Editing / On-Air", data=img_buf, file_name=f"custom_sky_{bearing:.0f}deg.png", mime="image/png")
        plt.close(fig)

import logging
import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")  # Safe headless execution for cloud servers
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import io
import numpy as np

# Core astronomical math engine
from skyfield.api import wgs84, Star, Loader
from skyfield.data import hipparcos
from skyfield.magnitudelib import planetary_magnitude

from astro import constellations, extinction, horizon, location, moon, sky_model

logger = logging.getLogger(__name__)

# Standard-atmosphere constants used to enable refraction for all altaz()
# calls below. This app has no live local weather input, so these are a
# reasonable fixed approximation -- still meaningfully more accurate near
# the horizon (this app's whole 0-40+deg window) than ignoring refraction
# entirely, which is what happened before.
STANDARD_TEMPERATURE_C = 10.0
STANDARD_PRESSURE_MBAR = 1010.0

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

    return ts, eph, stars_df


@st.cache_resource
def get_constellation_segments():
    return constellations.load_constellation_segments()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_sky_defaults(lat, lon):
    return location.suggest_sky_defaults(lat, lon)


@st.cache_data(ttl=3600, show_spinner="Sampling real terrain horizon...")
def cached_horizon_profile(lat, lon, az_min, az_max, n_samples=96):
    sample_az = np.linspace(az_min, az_max, n_samples)
    return sample_az, horizon.fetch_horizon_profile(lat, lon, sample_az)


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
    url_fov = np.clip(float(st.query_params.get("fov", 90.0)), 30.0, 180.0)
except ValueError:
    url_fov = 90.0
try:
    url_altmax = np.clip(float(st.query_params.get("altmax", 40.0)), 20.0, 90.0)
except ValueError:
    url_altmax = 40.0

DIRECTION_PRESETS = {
    "Custom": None, "North": 0.0, "Northeast": 45.0, "East": 90.0, "Southeast": 135.0,
    "South": 180.0, "Southwest": 225.0, "West": 270.0, "Northwest": 315.0,
}
preset_choice = st.sidebar.selectbox("Direction Preset", list(DIRECTION_PRESETS.keys()), index=0)

_seed_session_state_once("bearing", url_bearing)
if DIRECTION_PRESETS[preset_choice] is not None:
    st.session_state["bearing"] = DIRECTION_PRESETS[preset_choice]
_seed_session_state_once("fov_width", url_fov)
_seed_session_state_once("alt_max", url_altmax)

bearing = st.sidebar.slider("Compass Bearing (0=N, 90=E, 180=S, 270=W)", 0.0, 359.0, step=1.0, key="bearing")
fov_width = st.sidebar.slider("Field of View Width (deg)", 30.0, 180.0, step=5.0, key="fov_width")
alt_max = st.sidebar.slider("Max Altitude Shown (deg)", 20.0, 90.0, step=5.0, key="alt_max")

az_min = bearing - fov_width / 2.0
az_max = bearing + fov_width / 2.0

# --- SIDEBAR: 3. GRAPHIC TOGGLES ---
st.sidebar.header("3. Graphic Toggles")

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
show_labels = st.sidebar.checkbox("Show Planet/Moon Labels", value=True)
show_major_star_labels = st.sidebar.checkbox("Show Major Star Labels", value=True)
show_minor_star_labels = st.sidebar.checkbox("Show Minor Star Labels", value=False)
show_constellations = st.sidebar.checkbox("Show Constellation Lines", value=True)
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

# Persist the full view in the URL so a generated graphic is shareable/bookmarkable.
st.query_params["lat"] = f"{lat:.2f}"
st.query_params["lon"] = f"{lon:.2f}"
st.query_params["date"] = obs_date.isoformat()
st.query_params["time"] = obs_time.strftime("%H:%M")
st.query_params["tz"] = selected_tz
st.query_params["bearing"] = f"{bearing:.0f}"
st.query_params["fov"] = f"{fov_width:.0f}"
st.query_params["altmax"] = f"{alt_max:.0f}"
st.query_params["turbidity"] = f"{turbidity:.1f}"
st.query_params["brightness"] = f"{star_brightness:.1f}"


def size_for_magnitude(mag, reference_size=30.0, reference_mag=0.0, min_size=4.0, max_size=90.0):
    """Marker size from real apparent magnitude: each magnitude step is a
    ~2.5x brightness change, so size scales geometrically off a reference
    point rather than linearly, keeping both faint and very bright
    apparitions legible instead of vanishing or dominating the plot."""
    if mag is None or np.isnan(mag):
        return reference_size
    delta = reference_mag - mag
    return float(np.clip(reference_size * (1.35 ** delta), min_size, max_size))


# --- GRAPHIC GENERATION LOGIC ---
if st.button("Generate Sky Graphic", type="primary"):
    with st.spinner("Computing high-fidelity directional sky model and celestial structures..."):

        dt_local = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
        dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
        t = ts.from_datetime(dt_utc)
        observer_loc = earth + wgs84.latlon(lat, lon)

        sun_astrometric = observer_loc.at(t).observe(sun)
        sun_apparent = sun_astrometric.apparent()
        sun_alt, sun_az, _ = sun_apparent.altaz(temperature_C=STANDARD_TEMPERATURE_C, pressure_mbar=STANDARD_PRESSURE_MBAR)
        sun_deg = sun_alt.degrees
        sun_az_deg = sun_az.degrees

        fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100, facecolor='none')
        ax.set_xlim(az_min, az_max)
        ax.set_ylim(0, alt_max)

        # ======================================================================
        # SECTION 3: SKY BACKGROUND
        # Clear-daytime color comes from the Preetham analytic sky model
        # (astro/sky_model.py); twilight/night, which that model doesn't
        # cover, keep a hand-authored gradient + "Belt of Venus" treatment.
        # ======================================================================
        x_pixels, y_pixels = 150, 100
        x_space = np.linspace(az_min, az_max, x_pixels)
        y_space = np.linspace(0, alt_max, y_pixels)
        X_mesh, Y_mesh = np.meshgrid(x_space, y_space)

        rad_az_mesh = np.radians(X_mesh)
        rad_alt_mesh = np.radians(Y_mesh)
        rad_sun_az = np.radians(sun_az_deg)
        rad_sun_alt = np.radians(sun_deg)

        cos_scatter_angle = (np.sin(rad_alt_mesh) * np.sin(rad_sun_alt) +
                             np.cos(rad_alt_mesh) * np.cos(rad_sun_alt) * np.cos(rad_az_mesh - rad_sun_az))
        gamma_rad = np.arccos(np.clip(cos_scatter_angle, -1.0, 1.0))  # angular distance to the sun
        theta_rad = np.radians(90.0 - Y_mesh)  # zenith angle of each sky point

        day_sky_matrix = sky_model.daytime_sky_rgb(theta_rad, gamma_rad, sun_deg, turbidity)

        sun_alt_clamped = max(0.1, sun_deg)
        extinction_R, extinction_G, extinction_B = extinction.rgb_transmission(sun_alt_clamped, turbidity)

        # TURBIDITY 3: Calculate the global haze desaturation factor
        haze_blend = np.clip((turbidity - 1.0) / 4.0, 0, 1)

        base_navy = np.array([10, 16, 28]) / 255.0
        haze_navy = np.array([25, 30, 40]) / 255.0
        color_space_navy = base_navy * (1.0 - haze_blend) + haze_navy * haze_blend

        base_twilight = np.array([20, 45, 95]) / 255.0
        haze_twilight = np.array([45, 50, 60]) / 255.0
        color_twilight_base = base_twilight * (1.0 - haze_blend) + haze_twilight * haze_blend

        base_night = np.array([11, 17, 30]) / 255.0
        haze_night = np.array([20, 22, 28]) / 255.0
        color_night_base = base_night * (1.0 - haze_blend) + haze_night * haze_blend

        sun_filtered_R = 1.0 * extinction_R
        sun_filtered_G = 0.92 * extinction_G
        sun_filtered_B = 0.78 * extinction_B
        color_sunset_glow = np.array([sun_filtered_R, sun_filtered_G, sun_filtered_B])

        v_frac = Y_mesh / alt_max

        mie_width = 30.0 + (turbidity * 10.0)
        f_scatter = np.exp(-(np.degrees(gamma_rad) / mie_width) ** 2)

        twilight_horiz_glow = color_sunset_glow[None, None, :] * f_scatter[..., None] * (1.0 - v_frac[..., None]) * 0.90
        twilight_upper_sky = color_space_navy[None, None, :] * v_frac[..., None] + color_twilight_base[None, None, :] * (1.0 - v_frac[..., None])
        twilight_sky_matrix = np.clip(twilight_horiz_glow + twilight_upper_sky, 0, 1)

        night_sky_matrix = color_space_navy[None, None, :] * v_frac[..., None] + color_night_base[None, None, :] * (1.0 - v_frac[..., None])

        az_diff_rad = np.radians(X_mesh - sun_az_deg)
        h_shadow = np.degrees(np.arcsin(np.clip(np.sin(np.radians(sun_deg)) * np.cos(az_diff_rad), -1.0, 1.0)))

        belt_of_venus_mask = np.exp(-((Y_mesh - (h_shadow + 3.0)) / 3.0)**2) * np.clip((sun_deg + 5.0) / 5.0, 0, 1)
        belt_of_venus_mask = np.where(h_shadow < 0, belt_of_venus_mask, 0)

        twilight_sky_matrix[..., 0] += belt_of_venus_mask * 0.16
        twilight_sky_matrix[..., 1] += belt_of_venus_mask * 0.05
        twilight_sky_matrix[..., 2] += belt_of_venus_mask * 0.08

        # Perceptual lift for the hand-authored twilight/night gradients
        # (the day sky is already tone-mapped inside sky_model.daytime_sky_rgb).
        gamma_exponent = np.clip(1.0 + (sun_deg + 12.0) / 18.0, 1.0, 2.0)
        twilight_sky_matrix = np.clip(twilight_sky_matrix, 0.0, 1.0) ** (1.0 / gamma_exponent)
        night_sky_matrix = np.clip(night_sky_matrix, 0.0, 1.0) ** (1.0 / gamma_exponent)

        if sun_deg > 2.0:
            bg_image = day_sky_matrix
        elif sun_deg >= -2.0:
            fade_weight = np.clip((sun_deg + 2.0) / 4.0, 0, 1)
            bg_image = day_sky_matrix * fade_weight + twilight_sky_matrix * (1.0 - fade_weight)
        else:
            raw_fade = np.clip((sun_deg + 14.0) / 12.0, 0, 1)
            fade_twilight_to_night = np.power(raw_fade, 0.6)
            bg_image = twilight_sky_matrix * fade_twilight_to_night + night_sky_matrix * (1.0 - fade_twilight_to_night)

        bg_image = np.clip(bg_image, 0.0, 1.0)

        grid_color = "#ffffff" if sun_deg > 0 else ("#475569" if sun_deg > -6.0 else "#334155")

        ax.imshow(
            bg_image,
            extent=[az_min, az_max, 0, alt_max],
            origin="lower",
            aspect="auto",
            zorder=0
        )

        # --- ENGINE: MODERN LED HORIZONTAL CITY GLOW DOME ---
        if star_brightness <= 1.5 and sun_deg <= 0:
            x_glow, y_glow = 200, 100
            x_g_space = np.linspace(az_min, az_max, x_glow)
            y_g_space = np.linspace(0, alt_max, y_glow)
            X_m, Y_m = np.meshgrid(x_g_space, y_g_space)

            center_az = (az_min + az_max) / 2.0
            gaussian_glow = np.exp(-((X_m - center_az) / 70.0)**2 - (Y_m / 10.0)**2)
            glow_base_color = "#fbf8f0" if star_brightness == 1.0 else "#f4efe2"

            rgba_glow = np.zeros((y_glow, x_glow, 4))
            rgba_glow[..., :3] = matplotlib.colors.to_rgb(glow_base_color)
            rgba_glow[..., 3] = gaussian_glow * 0.28

            ax.imshow(
                rgba_glow, extent=[az_min, az_max, 0, alt_max], origin="lower",
                aspect="auto", zorder=1, interpolation="bilinear"
            )

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
                    mag = planetary_magnitude(astrometric)
                    size = size_for_magnitude(mag)
                    body_color = extinction.rgb_transmission(body_alt, turbidity)
                    ax.scatter(body_az, body_alt, s=size, color=body_color, zorder=50)
                    if show_labels:
                        ax.text(body_az + 0.5, body_alt + 0.5, label, color="#ffffff", fontsize=10, weight='bold', zorder=51)
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

            if az_min <= moon_az <= az_max and 0 <= moon_alt <= alt_max:
                m_pos = observer_loc.at(t).observe(moon_body).position.au
                s_pos = observer_loc.at(t).observe(sun).position.au

                m_dot_s = np.dot(m_pos, s_pos) / (np.linalg.norm(m_pos) * np.linalg.norm(s_pos))
                elongation = np.arccos(np.clip(m_dot_s, -1.0, 1.0))
                illuminated_fraction = 0.5 * (1.0 + np.cos(elongation))

                pabl_rad = moon.bright_limb_plot_angle_rad(sun_az_deg, sun_deg, moon_az, moon_alt)

                r_x = moon.angular_radius_deg(m_distance.km) * 2.5  # visually exaggerated for broadcast legibility
                r_y = r_x * (12.0 / 90.0) / (6.75 / alt_max)

                phi = np.linspace(-np.pi/2, np.pi/2, 30)

                x_outer_unit, y_outer_unit = np.cos(phi), np.sin(phi)

                phase_modifier = (illuminated_fraction - 0.5) * 2.0
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

                moon_patch = patches.PathPatch(moon_path, facecolor='#ffffff', edgecolor='none', zorder=50)
                ax.add_patch(moon_patch)

                if illuminated_fraction < 0.90:
                    dark_disk = patches.Ellipse((moon_az, moon_alt), width=r_x*2, height=r_y*2, facecolor='#ffffff', alpha=0.08, edgecolor='none', zorder=49)
                    ax.add_patch(dark_disk)

                if show_labels:
                    ax.text(moon_az + r_x + 0.4, moon_alt + 0.6, "Moon", color="#ffffff", fontsize=11, weight='bold', zorder=51)
        except Exception:
            logger.exception("Failed to compute/plot Moon position")

        # 5. DYNAMIC HIPPARCOS STAR FIELD
        if sun_deg <= -6:
            # Filter catalog natively via user slider
            visible_stars = stars_df[stars_df['magnitude'] <= star_brightness]
            star_obj = Star.from_dataframe(visible_stars)

            # Vectorized altitude/azimuth calculations for the entire visible catalog
            star_astrometric = observer_loc.at(t).observe(star_obj)
            s_alt, s_az, _ = star_astrometric.apparent().altaz(temperature_C=STANDARD_TEMPERATURE_C, pressure_mbar=STANDARD_PRESSURE_MBAR)

            star_az = s_az.degrees
            star_alt = s_alt.degrees

            star_az = normalize_az(star_az, az_min, az_max)

            # Cull stars strictly to viewport margins
            viewport_mask = (star_az >= az_min) & (star_az <= az_max) & (star_alt >= 0) & (star_alt <= alt_max)

            plot_az = star_az[viewport_mask]
            plot_alt = star_alt[viewport_mask]
            plot_mag = visible_stars['magnitude'].values[viewport_mask]
            plot_hips = visible_stars.index.values[viewport_mask]

            sizes = np.maximum(0.5, (5.0 - plot_mag) * 2.5)

            # Dim/redden stars near the horizon with the same atmospheric
            # extinction physics used for the sunset glow above.
            ext_r, ext_g, ext_b = extinction.rgb_transmission(plot_alt, turbidity)
            star_colors = np.stack([ext_r, ext_g, ext_b], axis=-1)

            # Plot the entire valid array simultaneously
            ax.scatter(plot_az, plot_alt, s=sizes, c=star_colors, alpha=0.7, zorder=20)

            # Dictionary of major anchor stars (Hipparcos ID -> Common Name)
            major_stars = {
                32349: "Sirius", 24608: "Capella", 69673: "Arcturus", 91262: "Vega",
                25336: "Rigel", 37279: "Procyon", 27989: "Betelgeuse", 97649: "Altair",
                21421: "Aldebaran", 65474: "Spica", 80112: "Antares", 37826: "Pollux",
                102098: "Deneb", 49669: "Regulus", 36850: "Castor", 11767: "Polaris"
            }

            if show_major_star_labels:
                for az_val, alt_val, hip_id in zip(plot_az, plot_alt, plot_hips):
                    if hip_id in major_stars:
                        ax.text(az_val + 0.4, alt_val + 0.4, major_stars[hip_id], color="#ffffff", fontsize=9, alpha=0.5, zorder=21)

            if show_minor_star_labels:
                for az_val, alt_val, hip_id in zip(plot_az, plot_alt, plot_hips):
                    if hip_id not in major_stars:
                        ax.text(az_val + 0.4, alt_val + 0.4, f"HIP {hip_id}", color="#ffffff", fontsize=7, alpha=0.35, zorder=21)

            # --- CONSTELLATION STICK FIGURES ---
            if show_constellations:
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

        # 6. HORIZON SILHOUETTE
        # Real terrain-derived horizon when the elevation lookup succeeds;
        # falls back to the original procedural "suburban trees" shape
        # (which is location-independent) if the API is slow/unavailable.
        x_silhouette_space = np.linspace(az_min, az_max, 400)
        try:
            sample_az, horizon_deg_samples = cached_horizon_profile(lat, lon, az_min, az_max)
            y_silhouette = np.clip(np.interp(x_silhouette_space, sample_az, horizon_deg_samples), 0.3, alt_max)
        except Exception:
            logger.exception("Terrain horizon lookup failed; falling back to procedural silhouette")
            base_ground = 4.0 + 1.0 * np.sin(x_silhouette_space / 5)
            tree_canopy = 1.2 * np.sin(x_silhouette_space * 2.5) * np.cos(x_silhouette_space * 0.4)
            fine_foliage = 0.5 * np.sin(x_silhouette_space * 12.0)
            y_silhouette = np.clip(base_ground + tree_canopy + fine_foliage, 2.0, min(10.0, alt_max))

        ax.fill_between(x_silhouette_space, -5, y_silhouette, color="#060c14", zorder=100)

        ax.grid(True, color=grid_color, alpha=0.15, linestyle='--', zorder=2)

        ax.set_xticks(np.arange(az_min, az_max + 1, 10))
        ax.set_yticks(np.arange(0, alt_max + 1, 10))

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=150, facecolor="none", edgecolor="none", pad_inches=0.0)
        img_buf.seek(0)

        st.download_button(label="💾 Download High-Res PNG for Editing / On-Air", data=img_buf, file_name=f"custom_sky_{bearing:.0f}deg.png", mime="image/png")
        plt.close(fig)

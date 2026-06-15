import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")  # Safe headless execution for cloud servers
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import numpy as np

# Core astronomical math engine
from skyfield.api import load, wgs84, Star

# Set page layout to wide for a clean dashboard feel
st.set_page_config(layout="wide", page_title="Custom Sky Graphic Generator")

st.title("🌌 Broadcast Sky Graphic Generator")
st.write("A lightweight, reliable engine rendering clean astronomical plates with native horizon silhouettes.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Observation Settings")
obs_date = st.sidebar.date_input("Select Date", datetime.now().date())

# Generate clean 12-hour time strings (AM/PM) with naive time object values
time_options = []
time_labels = []

for hour in range(24):
    for minute in [0, 15, 30, 45]:
        t_obj = time(hour, minute)
        time_options.append(t_obj)
        
        ampm_label = t_obj.strftime("%I:%M %p")
        time_labels.append(ampm_label)

# Set default index to 09:15 PM
selected_time_index = time_labels.index("09:15 PM") if "09:15 PM" in time_labels else 0

obs_time = st.sidebar.selectbox(
    "Select Time",
    options=time_options,
    format_func=lambda x: x.strftime("%I:%M %p"),
    index=selected_time_index
)

tz_options = ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]
selected_tz = st.sidebar.selectbox("Time Zone", tz_options, index=0)

# Location Input (Defaults to Central Ohio)
lat = st.sidebar.number_input("Latitude", value=40.00, step=0.01, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=-83.10, step=0.01, format="%.2f")

st.sidebar.header("2. View Window")
direction = st.sidebar.selectbox(
    "Looking Direction", 
    ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"], 
    index=6  # Default to West
)

# --- SIDEBAR: 3. GRAPHIC TOGGLES ---
st.sidebar.header("3. Graphic Toggles")

show_labels = st.sidebar.checkbox("Show Object Labels", value=True)
star_brightness = st.sidebar.slider("Star Visibility Limit", 1.0, 4.5, 2.5, step=0.5)

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

# Map all 8 headings to strict 90-degree rectangular Azimuth spans
az_map = {
    "North": (315, 405),
    "Northeast": (0, 90),
    "East": (45, 135),
    "Southeast": (90, 180),
    "South": (135, 225),
    "Southwest": (180, 270),
    "West": (225, 315),
    "Northwest": (270, 360)
}
az_min, az_max = az_map[direction]

# --- GRAPHIC GENERATION LOGIC ---
if st.button("Generate Sky Graphic", type="primary"):
    with st.spinner("Computing high-fidelity sky model and celestial structures..."):
        
        # Combine inputs into a localized datetime object
        dt_local = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
        dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
        
        # Initialize Skyfield ephemeris engines
        ts = load.timescale()
        t = ts.from_datetime(dt_utc)
        eph = load('de421.bsp')
        earth = eph['earth']
        observer_loc = earth + wgs84.latlon(lat, lon)
        
        # Calculate the sun's altitude and azimuth relative to your horizon
        sun = eph['sun']
        sun_astrometric = observer_loc.at(t).observe(sun)
        sun_alt, sun_az, _ = sun_astrometric.apparent().altaz()
        sun_deg = sun_alt.degrees
        
        # 1. ATMOSPHERIC GRADIENT CALCULATOR
        if sun_deg > 0:
            # Daytime sky blend model
            top_rgb = np.array([26, 102, 255]) / 255.0
            horizon_rgb = np.array([153, 204, 255]) / 255.0
            grid_color = "#ffffff"
            cmap_colors = [horizon_rgb, top_rgb]
        else:
            # Twilight & Night continuous equation mapping based on solar dip angles
            solar_dip = np.clip(abs(sun_deg), 0, 18)
            
            # Continuous factor mappings for top and base sky components
            t_factor = np.clip((solar_dip / 12.0), 0, 1)
            h_factor = np.clip((solar_dip / 8.0), 0, 1)
            
            top_dusk = np.array([15, 23, 42]) / 255.0
            top_night = np.array([11, 17, 32]) / 255.0
            top_rgb = top_dusk * (1.0 - t_factor) + top_night * t_factor
            
            horiz_twilight = np.array([212, 138, 59]) / 255.0  
            horiz_night = np.array([22, 34, 56]) / 255.0       
            horizon_rgb = horiz_twilight * (1.0 - h_factor) + horiz_night * h_factor
            
            if solar_dip < 6.0:
                mid_factor = solar_dip / 6.0
                mid_twilight = np.array([59, 45, 84]) / 255.0  
                mid_rgb = mid_twilight * (1.0 - mid_factor) + horiz_night * mid_factor
                cmap_colors = [horizon_rgb, mid_rgb, top_rgb]
                grid_color = "#475569"
            else:
                cmap_colors = [horizon_rgb, top_rgb]
                grid_color = "#334155"

        # 2. INITIALIZE MATPLOTLIB CANVAS
        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.set_xlim(az_min, az_max)
        ax.set_ylim(0, 40)
        
        # 3. RENDER THE ATMOSPHERIC COLOR GRADIENT BACKGROUND
        cmap = LinearSegmentedColormap.from_list("sky_gradient", cmap_colors)
        gradient_matrix = np.linspace(0, 1, 256).reshape(-1, 1)
        
        ax.imshow(
            gradient_matrix,
            extent=[az_min, az_max, 0, 40], 
            cmap=cmap,
            origin="lower",
            aspect="auto",
            zorder=0                        
        )
        
        # 4. PLOT PLANETS & MOON
        bodies = {
            'moon': (eph['moon'], 180, '🌙 Moon'),
            'mercury': (eph['mercury'], 40, 'Mercury'),
            'venus': (eph['venus'], 70, 'Venus'),
            'mars': (eph['mars'], 40, 'Mars'),
            'jupiter':

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
    with st.spinner("Computing high-fidelity directional sky model and celestial structures..."):
        
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
        sun_az_deg = sun_az.degrees
        
        # 2. INITIALIZE MATPLOTLIB CANVAS
        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.set_xlim(az_min, az_max)
        ax.set_ylim(0, 40)
        
        # 3. HIGH-FIDELITY 2D DIRECTIONAL ATMOSPHERIC MESH ENGINE
        x_pixels, y_pixels = 150, 100
        x_space = np.linspace(az_min, az_max, x_pixels)
        y_space = np.linspace(0, 40, y_pixels)
        X_mesh, Y_mesh = np.meshgrid(x_space, y_space)
        
        if sun_deg > 0:
            # Daytime Sky: Rich atmospheric blue
            grid_color = "#ffffff"
            bg_image = np.zeros((y_pixels, x_pixels, 3))
            top_rgb = np.array([26, 102, 255]) / 255.0
            horiz_rgb = np.array([153, 204, 255]) / 255.0
            for y in range(y_pixels):
                frac = y / float(y_pixels)
                bg_image[y, :, :] = horiz_rgb * (1.0 - frac) + top_rgb * frac
        else:
            # Twilight & Night: Compute dynamic light scattering paths based on look direction
            solar_dip = np.clip(abs(sun_deg), 0, 18)
            
            # Base color profiles
            top_dusk = np.array([15, 23, 42]) / 255.0
            top_night = np.array([11, 17, 32]) / 255.0
            horiz_twilight = np.array([212, 138, 59]) / 255.0  
            horiz_night = np.array([22, 34, 56]) / 255.0       
            mid_twilight = np.array([59, 45, 84]) / 255.0      
            
            t_factor = np.clip((solar_dip / 12.0), 0, 1)
            base_top_rgb = top_dusk * (1.0 - t_factor) + top_night * t_factor
            grid_color = "#475569" if solar_dip < 6.0 else "#334155"
            
            bg_image = np.zeros((y_pixels, x_pixels, 3))
            
            for y_idx in range(y_pixels):
                alt_val = y_space[y_idx]
                v_frac = alt_val / 40.0  
                
                for x_idx in range(x_pixels):
                    az_val = x_space[x_idx]
                    
                    # Normalize azimuth differences across the 360 degree boundary
                    az_diff = abs((az_val % 360) - (sun_az_deg % 360))
                    if az_diff > 180:
                        az_diff = 360 - az_diff
                        
                    # Horizontal scatter math: drops off smoothly the further away you look from the sunset
                    h_scatter = np.exp(-(az_diff / 45.0)**2)
                    
                    effective_dip = solar_dip + (az_diff / 10.0)
                    effective_dip = np.clip(effective_dip, 0, 18)
                    
                    h_factor = np.clip((effective_dip / 8.0), 0, 1)
                    local_horiz_rgb = horiz_twilight * (1.0 - h_factor) * h_scatter + horiz_night * (1.0 - (1.0 - h_factor) * h_scatter)
                    
                    if effective_dip < 6.0:
                        m_factor = effective_dip / 6.0
                        local_mid_rgb = mid_twilight * (1.0 - m_factor) * h_scatter + horiz_night * (1.0 - (1.0 - m_factor) * h_scatter)
                        
                        if v_frac < 0.35:
                            pixel_rgb = local_horiz_rgb * (1.0 - (v_frac / 0.35)) + local_mid_rgb * (v_frac / 0.35)
                        else:
                            p_frac = (v_frac - 0.35) / 0.65
                            pixel_rgb = local_mid_rgb * (1.0 - p_frac) + base_top_rgb * p_frac
                    else:
                        pixel_rgb = local_horiz_rgb * (1.0 - v_frac) + base_top_rgb * v_frac
                        
                    bg_image[y_idx, x_idx, :] = np.clip(pixel_rgb, 0, 1)

        # Draw the calculated 2D analytical gradient mesh onto the background plane
        ax.imshow(
            bg_image,
            extent=[az_min, az_max, 0, 40],
            origin="lower",
            aspect="auto",
            zorder=0
        )
        
        # --- ENGINE: URBAN SKY GLOW DOME IMAGE OVERLAY ---
        if star_brightness <= 1.5 and sun_deg <= 0:

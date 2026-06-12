import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")  # Safe headless execution for cloud servers
import matplotlib.pyplot as plt
import io
import numpy as np

# Core astronomical math engine
from skyfield.api import load, wgs84
from skyfield.named_stars import named_stars

# Set page layout to wide for a clean dashboard feel
st.set_page_config(layout="wide", page_title="Custom Sky Graphic Generator")

st.title("🌌 Broadcast Sky Graphic Generator")
st.write("A lightweight, reliable engine rendering clean astronomical plates with native horizon silhouettes.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Observation Settings")
obs_date = st.sidebar.date_input("Select Date", datetime.now().date())
obs_time = st.sidebar.time_input("Select Time", time(21, 15))  
tz_options = ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]
selected_tz = st.sidebar.selectbox("Time Zone", tz_options, index=0)

# Location Input (Defaults to Central Ohio)
lat = st.sidebar.number_input("Latitude", value=40.00, step=0.01, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=-83.10, step=0.01, format="%.2f")

st.sidebar.header("2. View Window")
direction = st.sidebar.selectbox(
    "Looking Direction", 
    ["East (Rising)", "West (Setting)", "South", "North"], 
    index=1  # Default to West
)

# Text label controls for easy cloning/editing workflows
st.sidebar.header("3. Graphic Toggles")
show_labels = st.sidebar.checkbox("Show Object Labels", value=True)
star_brightness = st.sidebar.slider("Star Visibility Limit (Lower magnitude = fewer stars)", 1.0, 4.5, 2.5, step=0.5)

# Map looking direction to strict rectangular Azimuth spans
az_map = {
    "East (Rising)": (45, 135),
    "West (Setting)": (225, 315),
    "South": (135, 225),
    "North": (315, 405)  # Shifts North across the 360 line linearly
}
az_min, az_max = az_map[direction]

# --- GRAPHIC GENERATION LOGIC ---
if st.button("Generate Sky Graphic", type="primary"):
    with st.spinner("Calculating orbital ephemerides..."):
        
        # Combine inputs into a localized datetime object
        dt_combined = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
        
        # Initialize Skyfield ephemeris engines
        ts = load.timescale()
        t = ts.from_datetime(dt_combined)
        eph = load('de421.bsp')
        earth = eph['earth']
        observer_loc = earth + wgs84.latlon(lat, lon)
        
        # 1. CALCULATE DYNAMIC BACKGROUND COLOR (Via Sun Position)
        sun = eph['sun']
        sun_alt, _, _ = observer_loc.at(t).observe(sun).apparent().altaz()
        sun_deg = sun_alt.degrees
        
        if sun_deg > 0:
            sky_color = "#1a75ff"       # Daytime Blue
            grid_color = "#66a3ff"
        elif sun_deg > -6:
            sky_color = "#1d2d44"       # Twilight Blue
            grid_color = "#415a77"
        else:
            sky_color = "#0c1821"       # Deep Night Void
            grid_color = "#1d3557"

        # 2. INITIALIZE STANDARD MATPLOTLIB CANVAS
        fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=sky_color)
        ax.set_facecolor(sky_color)
        
        # Force strict bounding coordinate limits (Altitude locked from 0° to 40° high)
        ax.set_xlim(az_min, az_max)
        ax.set_ylim(0, 40)
        
        # 3. PLOT PLANETS & MOON
        # List of major prominent bodies to track
        bodies = {
            'moon': (eph['moon'], 180, '🌙'),
            'mercury': (eph['mercury'], 40, 'Mercury'),
            'venus': (eph['venus'], 70, 'Venus'),
            'mars': (eph['mars'], 40, 'Mars'),
            'jupiter': (eph['jupiter'], 90, 'Jupiter'),
            'saturn': (eph['saturn'], 60, 'Saturn')
        }
        
        for name, (body, size, label) in bodies.items():
            astrometric = observer_loc.at(t).observe(body)
            alt, az, _ = astrometric.apparent().altaz()
            
            body_az = az.degrees
            body_alt = alt.degrees
            
            # Adjust North coordinates to fit our linear wraparound axis
            if direction == "North" and body_az < 90:
                body_az += 360
                
            # Render if it falls within the current viewing screen window
            if az_min <= body_az <= az_max and 0 <= body_alt <= 40:
                # Plot object marker
                color = "#ffffff" if name in ['moon', 'venus', 'jupiter'] else "#ff9999"
                ax.scatter(body_az, body_alt, s=size, color=color, zorder=50)
                
                # Render text tags if toggled on
                if show_labels:
                    ax.text(body_az + 1, body_alt + 1, label, color="#ffffff", fontsize=12, weight='bold', zorder=

import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")  # Safe headless execution for cloud servers
import matplotlib.pyplot as plt
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
        # FIXED: Targets are re-mapped to use Skyfield's direct kernel attributes smoothly
        bodies = {
            'moon': (eph['moon'], 180, '🌙 Moon'),
            'mercury': (eph['mercury'], 40, 'Mercury'),
            'venus': (eph['venus'], 70, 'Venus'),
            'mars': (eph['mars'], 40, 'Mars'),
            'jupiter': (eph['jupiter_barycenter'], 90, 'Jupiter'),
            'saturn': (eph['saturn_barycenter'], 60, 'Saturn')
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
                color = "#ffffff" if name in ['moon', 'venus', 'jupiter'] else "#ff9999"
                ax.scatter(body_az, body_alt, s=size, color=color, zorder=50)
                
                if show_labels:
                    ax.text(body_az + 0.6, body_alt + 0.6, label, color="#ffffff", fontsize=11, weight='bold', zorder=51)

        # 4. PLOT TRUE CALCULATED NAVIGATIONAL STARS
        if sun_deg <= -6:
            # Verified catalog of the brightest stars using true celestial coordinates (RA/Dec)
            # This completely bypasses the need to read heavy external database files on reload
            star_data = [
                ("Polaris", 1.97, "02h31m49.1s", "+89d15m51s"),
                ("Vega", 0.03, "18h36m56.3s", "+38d47m01s"),
                ("Capella", 0.08, "05h16m41.4s", "+45d59m53s"),
                ("Arcturus", -0.05, "14h15m39.7s", "+19d10m57s"),
                ("Betelgeuse", 0.50, "05h55m10.3s", "+07d24m25s"),
                ("Procyon", 0.34, "07h39m18.1s", "+05d13m30s"),
                ("Pollux", 1.14, "07h45m18.9s", "+28d01m34s"),
                ("Castor", 1.58, "07h34m36.0s", "+31d53m18s"),
                ("Spica", 0.98, "13h25m11.6s", "-11d09m41s"),
                ("Altair", 0.76, "19h50m47.0s", "+08d52m06s"),
                ("Deneb", 1.25, "20h41m25.9s", "+45d16m49s"),
                ("Regulus", 1.36, "10h08m22.3s", "+11d58m02s")
            ]
            
            for name, mag, ra_str, dec_str in star_data:
                if mag <= star_brightness:
                    # Dynamically compute exact Alt/Az paths based on your location parameters
                    star_obj = Star(ra_hours=ra_str, dec_degrees=dec_str)
                    star_astrometric = observer_loc.at(t).observe(star_obj)
                    s_alt, s_az, _ = star_astrometric.apparent().altaz()
                    
                    star_az = s_az.degrees
                    star_alt = s_alt.degrees
                    
                    if direction == "North" and star_az < 90:
                        star_az += 360
                        
                    if az_min <= star_az <= az_max and 0 <= star_alt <= 40:
                        size = max(4, (5.0 - mag) * 6)
                        ax.scatter(star_az, star_alt, s=size, color="#ffffff", alpha=0.8, zorder=20)
                        if show_labels:
                            ax.text(star_az + 0.5, star_alt + 0.5, name, color="#ffffff", fontsize=9, alpha=0.6, zorder=21)

        # 5. NATIVE FOREGROUND SILHOUETTE ENGINE
        # This standard rectangular array runs directly across our established azimuth range
        x_space = np.linspace(az_min, az_max, 400)
        
        # Mathematical formulas modeling an opaque hardwood forest treeline canopy
        base_ground = 4.0 + 1.0 * np.sin(x_space / 5)
        tree_canopy = 1.2 * np.sin(x_space * 2.5) * np.cos(x_space * 0.4)
        fine_foliage = 0.5 * np.sin(x_space * 12.0)
        
        y_treeline = base_ground + tree_canopy + fine_foliage
        y_treeline = np.clip(y_treeline, 2.0, 10.0)  # Confine foliage tops safely under 10 degrees high
        
        # Solid ground block fill directly over the underlying coordinate planes
        ax.fill_between(
            x_space,
            -5,               # Deep anchor baseline
            y_treeline,       # Crest of tree peaks
            color="#060c14",  # Solid dark silhouette tone
            zorder=100        # Secure top layer priority
        )
        
        # Clean up gridline decorations and formatting
        ax.grid(True, color=grid_color, alpha=0.2, linestyle='--', zorder=2)
        ax.set_xlabel("Azimuth (Degrees)", color="#ffffff")
        ax.set_ylabel("Altitude (Degrees)", color="#ffffff")
        ax.tick_params(colors='#ffffff')
        
        # Hide standard outer spine borders for a cleaner presentation style
        for spine in ax.spines.values():
            spine.set_visible(False)

        # --- DISPLAY & DOWNLOAD ---
        st.pyplot(fig)
        
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
        img_buf.seek(0)
        
        st.download_button(
            label="💾 Download High-Res PNG for Editing / On-Air",
            data=img_buf,
            file_name=f"custom_sky_{direction.split()[0]}.png",
            mime="image/png"
        )

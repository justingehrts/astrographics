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
        bodies = {
            'moon': (eph['moon'], 180, '🌙 Moon'),
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
                color = "#ffffff" if name in ['moon', 'venus', 'jupiter'] else "#ff9999"
                ax.scatter(body_az, body_alt, s=size, color=color, zorder=50)
                
                if show_labels:
                    ax.text(body_az + 0.6, body_alt + 0.6, label, color="#ffffff", fontsize=11, weight='bold', zorder=51)

        # 4. PLOT PROMINENT BRIGHT NAVIGATIONAL STARS
        if sun_deg <= -6:
            # High-visibility star catalog mapping: (Name, Magnitude, Approximate Azimuth Offset, Base Altitude)
            # This handles clean vector positioning tailored directly to our seasonal view frames
            star_catalog = [
                ('Polaris', 2.0, 360.0, 40.0), ('Vega', 0.0, 75.0, 35.0), 
                ('Capella', 0.1, 330.0, 15.0), ('Arcturus', -0.05, 200.0, 38.0),
                ('Betelgeuse', 0.5, 255.0, 12.0), ('Procyon', 0.4, 240.0, 8.0), 
                ('Pollux', 1.1, 295.0, 22.0), ('Castor', 1.6, 297.0, 24.0),
                ('Spica', 1.0, 185.0, 25.0), ('Altair', 0.8, 95.0, 20.0),
                ('Deneb', 1.2, 60.0, 30.0), ('Regulus', 1.4, 245.0, 32.0)
            ]
            
            for name, mag, s_az, s_alt in star_catalog:
                if mag <= star_brightness:
                    # Adjust linear azimuth for northern boundaries if needed
                    if direction == "North" and s_az < 90:
                        s_az += 360
                        
                    if az_min <= s_az <= az_max and 0 <= s_alt <= 40:
                        size = max(4, (5.0 - mag) * 6)
                        ax.scatter(s_az, s_alt, s=size, color="#ffffff", alpha=0.8, zorder=20)
                        if show_labels:
                            ax.text(s_az + 0.5, s_alt + 0.5, name, color="#ffffff", fontsize=9, alpha=0.6, zorder=21)

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

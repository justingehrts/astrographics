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
        # FIXED: Removed the emoji wireframe fallback token from the Moon parameter string
        bodies = {
            'moon': (eph['moon'], 180, 'Moon'),
            'mercury': (eph['mercury'], 40, 'Mercury'),
            'venus': (eph['venus'], 70, 'Venus'),
            'mars': (eph['mars'], 40, 'Mars'),
            'jupiter': (eph['jupiter_barycenter'], 90, 'Jupiter'),
            'saturn': (eph['saturn_barycenter'], 60, 'Saturn')
        }
        
        for name, (body, size, label) in bodies.items():
            try:
                astrometric = observer_loc.at(t).observe(body)
                alt, az, _ = astrometric.apparent().altaz()
                
                body_az = az.degrees
                body_alt = alt.degrees
                
                if direction == "North" and body_az < 90:
                    body_az += 360
                    
                if az_min <= body_az <= az_max and 0 <= body_alt <= 40:
                    color = "#ffffff" if name in ['moon', 'venus', 'jupiter'] else "#ff9999"
                    ax.scatter(body_az, body_alt, s=size, color=color, zorder=50)
                    
                    if show_labels:
                        ax.text(body_az + 0.6, body_alt + 0.6, label, color="#ffffff", fontsize=11, weight='bold', zorder=51)
            except Exception:
                continue

        # 5. PLOT TRUE CALCULATED NAVIGATIONAL STARS
        if sun_deg <= -6:
            star_data = [
                ("Polaris", 1.97, (2, 31, 49.1), (89, 15, 51)),
                ("Vega", 0.03, (18, 36, 56.3), (38, 47, 1)),
                ("Capella", 0.08, (5, 16, 41.4), (45, 59, 53)),
                ("Arcturus", -0.05, (14, 15, 39.7), (19, 10, 57)),
                ("Betelgeuse", 0.50, (5, 55, 10.3), (7, 24, 25)),
                ("Procyon", 0.34, (7, 39, 18.1), (5, 13, 30)),
                ("Pollux", 1.14, (7, 45, 18.9), (28, 1, 34)),
                ("Castor", 1.58, (7, 34, 36.0), (31, 53, 18)),
                ("Spica", 0.98, (13, 25, 11.6), (-11, 9, 41)),
                ("Altair", 0.76, (19, 50, 47.0), (8, 52, 6)),
                ("Deneb", 1.25, (20, 41, 25.9), (45, 16, 49)),
                ("Regulus", 1.36, (10, 8, 22.3), (11, 58, 2))
            ]
            
            for name, mag, ra_tuple, dec_tuple in star_data:
                if mag <= star_brightness:
                    try:
                        star_obj = Star(ra_hours=ra_tuple, dec_degrees=dec_tuple)
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
                    except Exception:
                        continue

        # 6. FIXED SUBURBAN TREE HORIZON SILHOUETTE
        x_space = np.linspace(az_min, az_max, 400)
        base_ground = 4.0 + 1.0 * np.sin(x_space / 5)
        tree_canopy = 1.2 * np.sin(x_space * 2.5) * np.cos(x_space * 0.4)
        fine_foliage = 0.5 * np.sin(x_space * 12.0)
        y_silhouette = base_ground + tree_canopy + fine_foliage
        y_silhouette = np.clip(y_silhouette, 2.0, 10.0)

        # Draw tree canopy layer
        ax.fill_between(x_space, -5, y_silhouette, color="#060c14", zorder=100)
        
        # Clean up gridline decorations and formatting
        ax.grid(True, color=grid_color, alpha=0.15, linestyle='--', zorder=2)
        ax.set_xlabel("Azimuth (Degrees)", color="#ffffff")
        ax.set_ylabel("Altitude (Degrees)", color="#ffffff")
        ax.tick_params(colors='#ffffff')
        
        for spine in ax.spines.values():
            spine.set_visible(False)

        # --- DISPLAY & DOWNLOAD ---
        # 1. Render natively to the Streamlit dashboard web interface
        st.pyplot(fig)
        
        # 2. Extract raw binary streams from the active plot object
        img_buf = io.BytesIO()
        
        # FIXED: Forces the figure canvas background to transparent and overrides tight bbox clip calculations.
        # This locks the dynamic gradient image edge-to-edge on your downloaded PNG.
        fig.savefig(
            img_buf, 
            format="png", 
            dpi=150, 
            facecolor="none", 
            edgecolor="none",
            pad_inches=0.0
        )
        img_buf.seek(0)
        
        # 3. Trigger the browser downloader callback
        st.download_button(
            label="💾 Download High-Res PNG for Editing / On-Air",
            data=img_buf,
            file_name=f"custom_sky_{direction}.png",
            mime="image/png"
        )

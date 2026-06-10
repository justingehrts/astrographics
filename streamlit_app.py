import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")  # Safe headless execution for cloud servers
import matplotlib.pyplot as plt
import io
import numpy as np

# Starplot and Skyfield imports
from starplot import HorizonPlot, Observer, styles, _
from skyfield.api import load, wgs84

# Set page layout to wide for a better dashboard feel
st.set_page_config(layout="wide", page_title="Sky Graphic Generator")

st.title("🌌 Custom Horizon Sky Graphic Generator")
st.write("Generate clean, broadcast-ready local sky charts that dynamically adapt to daylight and twilight.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Observation Settings")

obs_date = st.sidebar.date_input("Select Date", datetime.now().date())
obs_time = st.sidebar.time_input("Select Time", time(5, 0))  # Default to 5:00 AM
tz_options = ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]
selected_tz = st.sidebar.selectbox("Time Zone", tz_options, index=0)

# Location Input (Defaults to Central Ohio coordinates)
lat = st.sidebar.number_input("Latitude", value=40.00, step=0.01, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=-83.10, step=0.01, format="%.2f")

st.sidebar.header("2. View Window")
direction = st.sidebar.selectbox(
    "Looking Direction", 
    ["East (Rising)", "West (Setting)", "South", "North"], 
    index=0
)

# Standardize continuous negative degrees for flawless wrap boundaries looking North
az_map = {
    "East (Rising)": (45, 135),
    "West (Setting)": (225, 315),
    "South": (135, 225),
    "North": (-45, 45)  
}
az_min, az_max = az_map[direction]

# Max altitude view slider
alt_max = st.sidebar.slider("Vertical Sky Span (Degrees High)", min_value=20, max_value=90, value=40)

# --- GRAPHIC GENERATION LOGIC ---
if st.button("Generate Sky Graphic", type="primary"):
    with st.spinner("Calculating solar position and planetary ephemerides..."):
        
        # Combine date and time into a localized datetime object
        dt_combined = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
        
        # 1. DYNAMIC DAYLIGHT ENGINE (Skyfield)
        ts = load.timescale()
        t = ts.from_datetime(dt_combined)
        eph = load('de421.bsp')
        earth, sun = eph['earth'], eph['sun']
        observer_loc = earth + wgs84.latlon(lat, lon)
        
        # Calculate the sun's altitude relative to your horizon
        sun_alt = observer_loc.at(t).observe(sun).apparent().altaz()[0].degrees
        
        # Determine background color based on sun angle
        if sun_alt > 0:
            sky_color = "#1a75ff"       # Crisp daytime broadcast blue
            grid_color = "#4d94ff"      # Lighter blue grids for daytime visibility
        elif sun_alt > -6:
            sky_color = "#1d2d44"       # Vibrant civil twilight blue
            grid_color = "#415a77"
        else:
            sky_color = "#0c1821"       # Deep nighttime sky
            grid_color = "#1d3557"

        # 2. STARPLOT STRUCTURAL SETUP
        observer = Observer(latitude=lat, longitude=lon, dt=dt_combined)
        
        plot_style = styles.PlotStyle().extend(styles.extensions.BLUE_NIGHT)
        plot_style.background_color = sky_color
        
        # Build the frame
        p = HorizonPlot(
            observer=observer,
            azimuth=(az_min, az_max),
            altitude=(0, alt_max),
            style=plot_style,
            resolution=1600,             
        )
        
        # Plot celestial objects
        # Only show stars if it's actually dark enough to see them (Sun below -6 degrees)
        if sun_alt <= -6:
            p.stars(where=[_.magnitude < 2.5], style__label__font_color="#ffffff", style__label__font_size=11)
            
        p.planets(style__label__font_color="#ffffff", style__label__font_size=13)
        p.moon(style__label__font_color="#ffffff", style__label__font_size=13)
        
        # 3. VERIFIED VECTOR SILHOUETTE ENGINE
        # This executes at the very end of the loop, forcing the trees onto the top canvas layer.
        ax = p.ax
        xmin, xmax = ax.get_xlim()
        
        # Build a dense rolling tree canvas
        x_az = np.linspace(xmin, xmax, 500)
        base_hills = 4.0 + 1.2 * np.sin(x_az / 6)
        tree_jaggedness = 1.8 * np.sin(x_az * 1.5) * np.cos(x_az * 0.4)
        fine_branches = 0.8 * np.sin(x_az * 10.0)
        
        y_alt = base_hills + tree_jaggedness + fine_branches
        y_alt = np.clip(y_alt, 2.0, 10.0)  # Consistently clamp to exactly 10 degrees high max
        
        # Draw the opaque vector line directly over the finished plot space
        ax.fill_between(
            x_az, 0, y_alt, 
            color="#050b14", 
            zorder=100,            # High z-order guarantees it sits on top of Starplot layers
            clip_on=False          # Bypasses axis coordinate boundaries
        )
        
        # Clean up gridline visibility to match our dynamic sky background colors
        ax.grid(True, color=grid_color, alpha=0.4)

        # Force tight layout formatting
        p.fig.set_tight_layout(True)

        # --- DISPLAY & DOWNLOAD ---
        st.pyplot(p.fig)
        
        img_buf = io.BytesIO()
        p.fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=150)
        img_buf.seek(0)
        
        st.download_button(
            label="💾 Download High-Res PNG for On-Air / Social",
            data=img_buf,
            file_name=f"sky_graphic_{obs_date}_{direction.split()[0]}.png",
            mime="image/png"
        )

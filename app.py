import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from starplot import HorizonPlot, Observer, styles

# Set page layout to wide for a better dashboard feel
st.set_page_config(layout="wide", page_title="Sky Graphic Generator")

st.title("🌌 Custom Horizon Sky Graphic Generator")
st.write("Generate broadcast-ready sky charts without installing local Python environments.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Observation Settings")

# Date & Time Inputs
obs_date = st.sidebar.date_input("Select Date", datetime.now().date())
obs_time = st.sidebar.time_input("Select Time", time(5, 0))  # Default to 5:00 AM
tz_options = ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]
selected_tz = st.sidebar.selectbox("Time Zone", tz_options, index=0)

# Location Input (Defaults to Columbus, OH coordinates)
lat = st.sidebar.number_input("Latitude", value=40.0, step=0.1, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=-83.1, step=0.1, format="%.2f")

st.sidebar.header("2. View Window")
# Common broadcast viewing directions
direction = st.sidebar.selectbox(
    "Looking Direction", 
    ["East (Rising)", "West (Setting)", "South", "North"], 
    index=0
)

# Map direction string to exact azimuth degrees
az_map = {
    "East (Rising)": (45, 135),
    "West (Setting)": (225, 315),
    "South": (135, 225),
    "North": (315, 45) # Note: crossing 360/0 requires custom handling in some tools, or just use 45-135 style spans
}
az_min, az_max = az_map[direction]

# Max altitude view slider
alt_max = st.sidebar.slider("Vertical Sky Span (Degrees High)", min_value=20, max_value=90, value=40)

# --- GRAPHIC GENERATION LOGIC ---
if st.button("Generate Sky Graphic", type="primary"):
    with st.spinner("Calculating planetary ephemerides..."):
        
        # Combine date and time into a localized datetime object
        dt_combined = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
        
        # Define the observer
        observer = Observer(
            latitude=lat,
            longitude=lon,
            dt=dt_combined
        )
        
        # Define clean, presentation-ready styling
        plot_style = styles.PlotStyle(
            background_color="#0c1821",  # Deep nighttime sky
            text_color="#ffffff",
            font_name="sans-serif",      # Matplotlib fallback font
        )
        
        # Create Starplot Horizon object
        # Using a fixed high-definition aspect ratio close to 16:9
        p = HorizonPlot(
            observer=observer,
            az_min=az_min,
            az_max=az_max,
            alt_min=0,
            alt_max=alt_max,
            style=plot_style,
            resolution=1200, 
        )
        
        # Plot elements using clean, geometric markers
        p.stars()
        p.planets()
        p.moon()
        
        # Inject the custom silhouette overlay onto the exposed Matplotlib axis
        ax = p.ax
        
        try:
            # If you host the silhouette image in your GitHub repo alongside app.py
            tree_silhouette = plt.imread("tree_line_silhouette.png")
            ax.imshow(
                tree_silhouette,
                extent=[az_min, az_max, 0, 7],  # Spans the selected direction, sits 7° high
                aspect="auto",
                zorder=10                        # Keeps trees in front of background celestial bodies
            )
        except FileNotFoundError:
            # Elegant mathematical fallback if image isn't uploaded yet
            import numpy as np
            x_az = np.linspace(az_min, az_max, 200)
            y_alt = 3.0 + 1.2 * np.sin(x_az / 4) + 0.4 * np.sin(x_az / 1.5)
            ax.fill_between(x_az, 0, y_alt, color="#04090e", zorder=10)
            st.caption("⚠️ Using vector silhouette fallback. Upload 'tree_line_silhouette.png' to use custom imagery.")

        # --- DISPLAY & DOWNLOAD ---
        # Pass the underlying figure object directly to Streamlit
        st.pyplot(p.fig)
        
        # Save to buffer for easy download right from the web page
        import io
        img_buf = io.BytesIO()
        p.fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=150)
        img_buf.seek(0)
        
        st.download_button(
            label="💾 Download High-Res PNG for On-Air / Social",
            data=img_buf,
            file_name=f"sky_graphic_{obs_date}_{direction.split()[0]}.png",
            mime="image/png"
        )

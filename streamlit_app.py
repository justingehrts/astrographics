import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")  # Safe headless execution for cloud servers
import matplotlib.pyplot as plt
import io

# Starplot imports
from starplot import HorizonPlot, Observer, styles

# Set page layout to wide for a better dashboard feel
st.set_page_config(layout="wide", page_title="Sky Graphic Generator")

st.title("🌌 Custom Horizon Sky Graphic Generator")
st.write("Generate clean, broadcast-ready local sky charts for social media and on-air graphics.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Observation Settings")

# Date & Time Inputs
obs_date = st.sidebar.date_input("Select Date", datetime.now().date())
obs_time = st.sidebar.time_input("Select Time", time(5, 0))  # Default to 5:00 AM pre-dawn
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

# FIXED: Standardizes negative degrees for seamless Matplotlib wrap boundaries looking North
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
    with st.spinner("Calculating planetary ephemerides and rendering..."):
        
        # Combine date and time into a localized datetime object
        dt_combined = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
        
        # Define the observer profile
        observer = Observer(
            latitude=lat,
            longitude=lon,
            dt=dt_combined
        )
        
        # FIXED: Initialize base style and properly attach a built-in Starplot extension map 
        plot_style = styles.PlotStyle().extend(
            styles.extensions.BLUE_NIGHT
        )
        
        # Safely tweak custom color variables using standard hex strings
        plot_style.background_color = "#0c1821"  # Deep sky background
        plot_style.text_color = "#ffffff"        # Clean white labels
        
        # Create Starplot Horizon object
        p = HorizonPlot(
            observer=observer,
            az_min=az_min,
            az_max=az_max,
            alt_min=0,
            alt_max=alt_max,
            style=plot_style,
            resolution=1600,             # Sharp pixel resolution for 16:9 sizing
        )
        
        # Plot standard elements using geometric styling
        p.stars()
        p.planets()
        p.moon()
        
        # Gain access to the underlying Matplotlib axis object
        ax = p.ax
        
        # Dynamically sample the boundaries directly from the plot coordinates
        xmin, xmax = ax.get_xlim()
        ymin = 0
        ymax = 10  # Always lock the height to exactly 10 degrees high
        
        try:
            # Look for the asset image in your root folder
            tree_silhouette = plt.imread("tree_line_silhouette.png")
            
            ax.imshow(
                tree_silhouette,
                extent=[xmin, xmax, ymin, ymax],
                aspect="auto",  # Stretch or compress horizontally to seamlessly fit the dynamic window
                zorder=10       # Keeps trees strictly in front of low-hanging planets/moon
            )
        except FileNotFoundError:
            # Procedural vector backup if your workspace asset file is missing
            import numpy as np
            x_az = np.linspace(xmin, xmax, 300)
            # Builds a quick organic hilly baseline resting between 3 to 5 degrees high
            y_alt = 4.0 + 1.0 * np.sin(x_az / 4) + 0.3 * np.sin(x_az / 1.5)
            ax.fill_between(x_az, 0, y_alt, color="#04090e", zorder=10)
            st.caption("⚠️ Using vector silhouette fallback. Upload 'tree_line_silhouette.png' to see custom tree imagery.")

        # FIXED: Ensure layout constraints prevent boundary labeling cutoffs
        p.fig.set_tight_layout(True)

        # --- DISPLAY & DOWNLOAD ---
        # Render the matplotlib plot cleanly right on the web application page
        st.pyplot(p.fig)
        
        # Convert figure to an in-memory byte stream for high-res downloading
        img_buf = io.BytesIO()
        p.fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=150)
        img_buf.seek(0)
        
        st.download_button(
            label="💾 Download High-Res PNG for On-Air / Social",
            data=img_buf,
            file_name=f"sky_graphic_{obs_date}_{direction.split()[0]}.png",
            mime="image/png"
        )

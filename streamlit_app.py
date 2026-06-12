import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo

# Set page layout to wide for a better dashboard feel
st.set_page_config(layout="wide", page_title="Stellarium Web Graphic Engine")

st.title("🌌 Interactive Sky Graphic Portal (Stellarium Web)")
st.write("An agile, zero-memory dashboard mapping your exact local coordinates directly into Stellarium's engine.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Observation Settings")

obs_date = st.sidebar.date_input("Select Date", datetime.now().date())
obs_time = st.sidebar.time_input("Select Time", time(21, 15))  # Default to mid-twilight
tz_options = ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]
selected_tz = st.sidebar.selectbox("Time Zone", tz_options, index=0)

# Location Input (Defaults to Central Ohio coordinates)
lat = st.sidebar.number_input("Latitude", value=40.00, step=0.01, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=-83.10, step=0.01, format="%.2f")

# Combine inputs into a localized datetime object to fetch the exact timestamp
dt_combined = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
iso_timestamp = dt_combined.strftime("%Y-%m-%dT%H:%M:%S")

# --- STELLARIUM WEB CONTEXT STRINGS ---
# We build standard coordinates strings to pass directly down to the frame
stellarium_url = (
    f"https://stellarium-web.org/"
    f"?lat={lat:.2f}"
    f"&lng={lon:.2f}"
    f"&date={iso_timestamp}"
    f"&fov=60.0"  # Sets a clean, standard 60-degree broad horizon field of view
)

# --- RENDERING THE WEBASSEMBLY ENGINE ---
st.subheader("📺 Live Capture Viewport")
st.write("Drag to change view directions. Use the toolbar buttons at the bottom to toggle landscapes, grids, or lines before screenshotting.")

# Injecting a responsive iframe frame container directly into your Streamlit canvas wrapper
st.components.v1.html(
    f"""
    <iframe 
        src="{stellarium_url}" 
        width="100%" 
        height="675" 
        style="border:none; border-radius:8px; background-color:#000000;"
        allow="geolocation">
    </iframe>
    """,
    height=690,
)

# --- SCREENSHOT WORKFLOW INSTRUCTIONS ---
st.markdown("---")
st.markdown("""
### 📸 Production Workflow Checklist
1. **Set the View:** Click and drag directly inside the window above to face **West** (or any other desired heading).
2. **Clean the Canvas:** Use the pop-up buttons at the very bottom center of the interactive map frame to turn off the atmosphere, constellation art, or ground layers as needed.
3. **Capture:** Snag your high-resolution screenshot directly from the browser window. The canvas is locked to a presentation-ready size so you can cleanly annotate it.
""")

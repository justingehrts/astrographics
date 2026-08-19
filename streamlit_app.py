import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")  # Safe headless execution for cloud servers
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
import matplotlib.patches as patches
import io
import numpy as np
import pandas as pd

# Core astronomical math engine
from skyfield.api import load, wgs84, Star, Loader
from skyfield.data import hipparcos

# Set page layout to wide for a clean dashboard feel
st.set_page_config(layout="wide", page_title="Custom Sky Graphic Generator")

st.title("🌌 Broadcast Sky Graphic Generator")
st.write("A lightweight, reliable engine rendering clean astronomical plates with native horizon silhouettes.")

# ======================================================================
# CACHED DATA ENGINE
# Loads the heavy ephemeris and star catalog files into memory exactly once,
# preventing timeouts and massive latency delays on cloud deployments.
# ======================================================================
@st.cache_resource
def get_astronomy_data():
    loader = Loader('.')
    ts = loader.timescale()
    eph = loader('de421.bsp')
    
    # FIXED: Replaced 'hip_main.dat' with hipparcos.URL
    # This forces Streamlit Cloud to download the catalog if it's missing natively.
    with loader.open(hipparcos.URL) as f:
        stars_df = hipparcos.load_dataframe(f)
        
    return ts, eph, stars_df

# Load the data models natively
ts, eph, stars_df = get_astronomy_data()
earth = eph['earth']
sun = eph['sun']

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

# ======================================================================
# URL QUERY PARAMETER ENGINE
# ======================================================================
try:
    url_lat = float(st.query_params.get("lat", 39.96))
except ValueError:
    url_lat = 39.96

try:
    url_lon = float(st.query_params.get("lon", -83.00))
except ValueError:
    url_lon = -83.00

lat = st.sidebar.number_input("Latitude", value=url_lat, step=0.01, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=url_lon, step=0.01, format="%.2f")

st.query_params["lat"] = f"{lat:.2f}"
st.query_params["lon"] = f"{lon:.2f}"

st.sidebar.header("2. View Window")
direction = st.sidebar.selectbox(
    "Looking Direction", 
    ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"], 
    index=6
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

az_map = {
    "North": (315, 405), "Northeast": (0, 90), "East": (45, 135), "Southeast": (90, 180),
    "South": (135, 225), "Southwest": (180, 270), "West": (225, 315), "Northwest": (270, 360)
}
az_min, az_max = az_map[direction]

# --- GRAPHIC GENERATION LOGIC ---
if st.button("Generate Sky Graphic", type="primary"):
    with st.spinner("Computing high-fidelity directional sky model and celestial structures..."):
        
        dt_local = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
        dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
        t = ts.from_datetime(dt_utc)
        observer_loc = earth + wgs84.latlon(lat, lon)
        
        sun_astrometric = observer_loc.at(t).observe(sun)
        sun_alt, sun_az, _ = sun_astrometric.apparent().altaz()
        sun_deg = sun_alt.degrees
        sun_az_deg = sun_az.degrees
                
        fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100, facecolor='none')
        ax.set_xlim(az_min, az_max)
        ax.set_ylim(0, 40)
        
        # ======================================================================
        # SECTION 3: STELLARIUM-ADAPTED VECTORIZED LIGHT PATH ENGINE
        # ======================================================================
        x_pixels, y_pixels = 150, 100
        x_space = np.linspace(az_min, az_max, x_pixels)
        y_space = np.linspace(0, 40, y_pixels)
        X_mesh, Y_mesh = np.meshgrid(x_space, y_space)
        
        rad_az_mesh = np.radians(X_mesh)
        rad_alt_mesh = np.radians(Y_mesh)
        rad_sun_az = np.radians(sun_az_deg)
        rad_sun_alt = np.radians(sun_deg)
        
        cos_scatter_angle = (np.sin(rad_alt_mesh) * np.sin(rad_sun_alt) + 
                             np.cos(rad_alt_mesh) * np.cos(rad_sun_alt) * np.cos(rad_az_mesh - rad_sun_az))
        theta_deg = np.degrees(np.arccos(np.clip(cos_scatter_angle, -1.0, 1.0)))
        
        f_scatter = np.exp(-(theta_deg / 50.0)**2)  
        
        sun_alt_clamped = max(0.1, sun_deg)
        air_mass_sun = 1.0 / (np.sin(np.radians(sun_alt_clamped)) + 0.15 * (sun_alt_clamped + 3.885) ** -1.253)
        
        extinction_R = np.exp(-0.02 * air_mass_sun)
        extinction_G = np.exp(-0.04 * air_mass_sun)
        extinction_B = np.exp(-0.10 * air_mass_sun)
        
        color_sky_blue = np.array([35, 115, 245]) / 255.0
        color_space_navy = np.array([10, 16, 28]) / 255.0
        
        sun_filtered_R = 1.0 * extinction_R
        sun_filtered_G = 0.92 * extinction_G
        sun_filtered_B = 0.78 * extinction_B
        color_sunset_glow = np.array([sun_filtered_R, sun_filtered_G, sun_filtered_B])
        
        v_frac = Y_mesh / 40.0
        
        day_progress = np.clip((sun_deg - 1.0) / 6.0, 0, 1)
        ambient_day_horiz = color_sky_blue * 0.4 * day_progress + color_sunset_glow * 0.85 * (1.0 - day_progress)
        
        day_base = ambient_day_horiz[None, None, :] * (1.0 - v_frac[..., None]) + color_sky_blue[None, None, :] * v_frac[..., None]
        day_mie_glow = color_sunset_glow[None, None, :] * f_scatter[..., None] * 0.4
        day_sky_matrix = np.clip(day_base + day_mie_glow, 0, 1)
        
        twilight_horiz_glow = color_sunset_glow[None, None, :] * f_scatter[..., None] * (1.0 - v_frac[..., None]) * 0.90
        twilight_upper_sky = color_space_navy[None, None, :] * v_frac[..., None] + np.array([20, 45, 95])[None, None, :] / 255.0 * (1.0 - v_frac[..., None])
        twilight_sky_matrix = np.clip(twilight_horiz_glow + twilight_upper_sky, 0, 1)
        
        night_sky_matrix = color_space_navy[None, None, :] * v_frac[..., None] + np.array([11, 17, 30]) / 255.0 * (1.0 - v_frac[..., None])
        
        az_diff_rad = np.radians(X_mesh - sun_az_deg)
        h_shadow = np.degrees(np.arcsin(np.clip(np.sin(np.radians(sun_deg)) * np.cos(az_diff_rad), -1.0, 1.0)))
        
        belt_of_venus_mask = np.exp(-((Y_mesh - (h_shadow + 3.0)) / 3.0)**2) * np.clip((sun_deg + 5.0) / 5.0, 0, 1)
        belt_of_venus_mask = np.where(h_shadow < 0, belt_of_venus_mask, 0)
        
        twilight_sky_matrix[..., 0] += belt_of_venus_mask * 0.16  
        twilight_sky_matrix[..., 1] += belt_of_venus_mask * 0.05
        twilight_sky_matrix[..., 2] += belt_of_venus_mask * 0.08
        
        if sun_deg > 2.0:
            bg_image = day_sky_matrix
        elif sun_deg >= -2.0:
            fade_weight = np.clip((sun_deg + 2.0) / 4.0, 0, 1)
            bg_image = day_sky_matrix * fade_weight + twilight_sky_matrix * (1.0 - fade_weight)
        else:
            raw_fade = np.clip((sun_deg + 14.0) / 12.0, 0, 1)
            fade_twilight_to_night = np.power(raw_fade, 0.6)
            bg_image = twilight_sky_matrix * fade_twilight_to_night + night_sky_matrix * (1.0 - fade_twilight_to_night)
            
        gamma_exponent = np.clip(1.0 + (sun_deg + 12.0) / 18.0, 1.0, 2.0)
        
        bg_image = np.clip(bg_image, 0.0, 1.0)
        bg_image = bg_image ** (1.0 / gamma_exponent)
        
        grid_color = "#ffffff" if sun_deg > 0 else ("#475569" if sun_deg > -6.0 else "#334155")

        ax.imshow(
            bg_image,
            extent=[az_min, az_max, 0, 40],
            origin="lower",
            aspect="auto",
            zorder=0
        )
        
        # --- ENGINE: MODERN LED HORIZONTAL CITY GLOW DOME ---
        if star_brightness <= 1.5 and sun_deg <= 0:
            x_glow, y_glow = 200, 100
            x_g_space = np.linspace(az_min, az_max, x_glow)
            y_g_space = np.linspace(0, 40, y_glow)
            X_m, Y_m = np.meshgrid(x_g_space, y_g_space)
            
            center_az = (az_min + az_max) / 2.0
            gaussian_glow = np.exp(-((X_m - center_az) / 70.0)**2 - (Y_m / 10.0)**2)
            glow_base_color = "#fbf8f0" if star_brightness == 1.0 else "#f4efe2"
            
            rgba_glow = np.zeros((y_glow, x_glow, 4))
            rgba_glow[..., :3] = matplotlib.colors.to_rgb(glow_base_color)
            rgba_glow[..., 3] = gaussian_glow * 0.28  
            
            ax.imshow(
                rgba_glow, extent=[az_min, az_max, 0, 40], origin="lower",
                aspect="auto", zorder=1, interpolation="bilinear"
            )

        # 4. PLOT PLANETS & DYNAMIC MOON ENGINE
        bodies = {
            'mercury': (eph['mercury'], 16, 'Mercury'),  
            'venus': (eph['venus'], 35, 'Venus'),      
            'mars': (eph['mars'], 16, 'Mars'),         
            'jupiter': (eph['jupiter_barycenter'], 45, 'Jupiter'), 
            'saturn': (eph['saturn_barycenter'], 24, 'Saturn')     
        }
        
        for name, (body, size, label) in bodies.items():
            try:
                astrometric = observer_loc.at(t).observe(body)
                alt, az, _ = astrometric.apparent().altaz()
                
                body_az, body_alt = az.degrees, alt.degrees
                
                if direction == "North" and body_az < 90:
                    body_az += 360
                    
                if az_min <= body_az <= az_max and 0 <= body_alt <= 40:
                    ax.scatter(body_az, body_alt, s=size, color="#ffffff", zorder=50)
                    if show_labels:
                        ax.text(body_az + 0.5, body_alt + 0.5, label, color="#ffffff", fontsize=10, weight='bold', zorder=51)
            except Exception:
                continue

        # --- DYNAMIC ROTATIONAL MOON PHASE VECTOR PATH ENGINE ---
        try:
            moon_body = eph['moon']
            moon_astrometric = observer_loc.at(t).observe(moon_body)
            m_alt, m_az, _ = moon_astrometric.apparent().altaz()
            
            moon_az, moon_alt = m_az.degrees, m_alt.degrees
            
            if direction == "North" and moon_az < 90:
                moon_az += 360
                
            if az_min <= moon_az <= az_max and 0 <= moon_alt <= 40:
                m_pos = observer_loc.at(t).observe(moon_body).position.au
                s_pos = observer_loc.at(t).observe(sun).position.au
                
                m_dot_s = np.dot(m_pos, s_pos) / (np.linalg.norm(m_pos) * np.linalg.norm(s_pos))
                elongation = np.arccos(np.clip(m_dot_s, -1.0, 1.0))
                illuminated_fraction = 0.5 * (1.0 + np.cos(elongation))
                
                pabl_rad = np.arctan2(sun_alt.degrees - moon_alt, sun_az_deg - moon_az)
                
                r_x = 0.65  
                r_y = r_x * (12.0 / 90.0) / (6.75 / 40.0)
                
                phi = np.linspace(-np.pi/2, np.pi/2, 30)
                
                x_outer_unit, y_outer_unit = np.cos(phi), np.sin(phi)
                
                phase_modifier = (illuminated_fraction - 0.5) * 2.0
                x_inner_unit, y_inner_unit = x_outer_unit * phase_modifier, y_outer_unit
                
                cos_p, sin_p = np.cos(pabl_rad), np.sin(pabl_rad)
                
                # VECTORIZED MOON TERMINATOR MATH
                x_out_rot = x_outer_unit * cos_p - y_outer_unit * sin_p
                y_out_rot = x_outer_unit * sin_p + y_outer_unit * cos_p
                
                x_in_rot = (x_inner_unit * cos_p - y_inner_unit * sin_p)[::-1]
                y_in_rot = (x_inner_unit * sin_p + y_inner_unit * cos_p)[::-1]
                
                x_verts = np.concatenate([x_out_rot, x_in_rot]) * r_x + moon_az
                y_verts = np.concatenate([y_out_rot, y_in_rot]) * r_y + moon_alt
                
                verts = np.column_stack((x_verts, y_verts))
                verts = np.vstack((verts, verts[0])) # Close polygon
                
                codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
                moon_path = Path(verts, codes)
                
                moon_patch = patches.PathPatch(moon_path, facecolor='#ffffff', edgecolor='none', zorder=50)
                ax.add_patch(moon_patch)
                
                if illuminated_fraction < 0.90:
                    dark_disk = patches.Ellipse((moon_az, moon_alt), width=r_x*2, height=r_y*2, facecolor='#ffffff', alpha=0.08, edgecolor='none', zorder=49)
                    ax.add_patch(dark_disk)
                    
                if show_labels:
                    ax.text(moon_az + r_x + 0.4, moon_alt + 0.6, "Moon", color="#ffffff", fontsize=11, weight='bold', zorder=51)
        except Exception:
            pass

        # 5. DYNAMIC HIPPARCOS STAR FIELD
        if sun_deg <= -6:
            # Filter catalog natively via user slider
            visible_stars = stars_df[stars_df['magnitude'] <= star_brightness]
            star_obj = Star.from_dataframe(visible_stars)
            
            # Vectorized altitude/azimuth calculations for the entire visible catalog
            star_astrometric = observer_loc.at(t).observe(star_obj)
            s_alt, s_az, _ = star_astrometric.apparent().altaz()
            
            star_az = s_az.degrees
            star_alt = s_alt.degrees
            
            if direction == "North":
                star_az = np.where(star_az < 90, star_az + 360, star_az)
                
            # Cull stars strictly to viewport margins
            viewport_mask = (star_az >= az_min) & (star_az <= az_max) & (star_alt >= 0) & (star_alt <= 40)
            
            plot_az = star_az[viewport_mask]
            plot_alt = star_alt[viewport_mask]
            plot_mag = visible_stars['magnitude'].values[viewport_mask]
            plot_hips = visible_stars.index.values[viewport_mask]
            
            sizes = np.maximum(0.5, (5.0 - plot_mag) * 2.5)
            
            # Plot the entire valid array simultaneously
            ax.scatter(plot_az, plot_alt, s=sizes, color="#ffffff", alpha=0.7, zorder=20)
            
            # Dictionary of major anchor stars (Hipparcos ID -> Common Name)
            major_stars = {
                32349: "Sirius", 24608: "Capella", 69673: "Arcturus", 91262: "Vega", 
                25336: "Rigel", 37279: "Procyon", 27989: "Betelgeuse", 97649: "Altair", 
                21421: "Aldebaran", 65474: "Spica", 80112: "Antares", 37826: "Pollux", 
                102098: "Deneb", 49669: "Regulus", 36850: "Castor", 677: "Polaris"
            }
            
            if show_labels:
                for az_val, alt_val, hip_id in zip(plot_az, plot_alt, plot_hips):
                    if hip_id in major_stars:
                        ax.text(az_val + 0.4, alt_val + 0.4, major_stars[hip_id], color="#ffffff", fontsize=9, alpha=0.5, zorder=21)

        # 6. FIXED SUBURBAN TREE HORIZON SILHOUETTE
        x_silhouette_space = np.linspace(az_min, az_max, 400)
        base_ground = 4.0 + 1.0 * np.sin(x_silhouette_space / 5)
        tree_canopy = 1.2 * np.sin(x_silhouette_space * 2.5) * np.cos(x_silhouette_space * 0.4)
        fine_foliage = 0.5 * np.sin(x_silhouette_space * 12.0)
        y_silhouette = np.clip(base_ground + tree_canopy + fine_foliage, 2.0, 10.0)

        ax.fill_between(x_silhouette_space, -5, y_silhouette, color="#060c14", zorder=100)
        
        ax.grid(True, color=grid_color, alpha=0.15, linestyle='--', zorder=2)
        
        ax.set_xticks(np.arange(az_min, az_max + 1, 10))
        ax.set_yticks(np.arange(0, 41, 10))
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)
        
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=150, facecolor="none", edgecolor="none", pad_inches=0.0)
        img_buf.seek(0)
        
        st.download_button(label="💾 Download High-Res PNG for Editing / On-Air", data=img_buf, file_name=f"custom_sky_{direction}.png", mime="image/png")
        plt.close(fig)

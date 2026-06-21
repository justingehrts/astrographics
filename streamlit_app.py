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
                
        # FORCE FIXED RENDERING DPI TO GUARANTEE TEXT & DOT SCALING CONSISTENCY
        fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100, facecolor='none')
        ax.set_xlim(az_min, az_max)
        ax.set_ylim(0, 40)
        
        # ======================================================================
        # SECTION 3: STELLARIUM-ADAPTED VECTORIZED LIGHT PATH ENGNE
        # Computes continuous Rayleigh/Mie scattering profiles without step limits.
        # ======================================================================
        x_pixels, y_pixels = 150, 100
        x_space = np.linspace(az_min, az_max, x_pixels)
        y_space = np.linspace(0, 40, y_pixels)
        X_mesh, Y_mesh = np.meshgrid(x_space, y_space)
        
        # Convert geometric angles to true radians for 2D matrix calculation
        rad_az_mesh = np.radians(X_mesh)
        rad_alt_mesh = np.radians(Y_mesh)
        rad_sun_az = np.radians(sun_az_deg)
        rad_sun_alt = np.radians(sun_deg)
        
        # Compute exact angular scattering distance (theta) across the canvas frame
        cos_scatter_angle = (np.sin(rad_alt_mesh) * np.sin(rad_sun_alt) + 
                             np.cos(rad_alt_mesh) * np.cos(rad_sun_alt) * np.cos(rad_az_mesh - rad_sun_az))
        theta_deg = np.degrees(np.arccos(np.clip(cos_scatter_angle, -1.0, 1.0)))
        
        # Calculate localized scattering intensity parameters natively
        f_scatter = np.exp(-(theta_deg / 50.0)**2)  # Mie forward scattering glow dome
        
        # 1. COMPUTE CONTINUOUS AIR MASS PATHWAY & ATMOSPHERIC EXTINCTION
        # Simulates the physical thickening of air filters as the sun sets
        sun_alt_clamped = max(0.1, sun_deg)
        air_mass_sun = 1.0 / (np.sin(np.radians(sun_alt_clamped)) + 0.15 * (sun_alt_clamped + 3.885) ** -1.253)
        
        # Wavelength absorption scaling vectors (Rayleigh Extinction profiles)
        # Red wavelengths pass cleanly; blue wavelengths are completely scattered away at low angles
        extinction_R = np.exp(-0.02 * air_mass_sun)
        extinction_G = np.exp(-0.04 * air_mass_sun)
        extinction_B = np.exp(-0.10 * air_mass_sun)
        
        # 2. DEFINE PHYSICAL SCATTERING COLOR BASES
        color_sky_blue = np.array([30, 110, 255]) / 255.0
        color_space_navy = np.array([10, 16, 28]) / 255.0
        
        # The sun's light shifts dynamically down from bright white to deep gold/amber based on air mass
        sun_filtered_R = 1.0 * extinction_R
        sun_filtered_G = 0.95 * extinction_G
        sun_filtered_B = 0.82 * extinction_B
        color_sunset_glow = np.array([sun_filtered_R, sun_filtered_G, sun_filtered_B])
        
        # Calculate global illumination factor
        illumination = np.clip((sun_deg + 6.0) / 12.0, 0, 1)
        
        # 3. VECTORIZED SKY MESH BUILDING
        v_frac = Y_mesh / 40.0
        
        # Generate the daytime Rayleigh sky base matrix
        day_base = color_sky_blue[None, None, :] * (1.0 - v_frac[..., None]) + (color_sky_blue * 0.4)[None, None, :] * v_frac[..., None]
        day_mie_glow = color_sunset_glow[None, None, :] * f_scatter[..., None] * 0.4
        day_sky_matrix = np.clip(day_base + day_mie_glow, 0, 1)
        
        # Generate the twilight scattering base matrix (glowing bronze horizon fading up to deep indigo)
        twilight_horiz_glow = color_sunset_glow[None, None, :] * f_scatter[..., None] * np.exp(-v_frac * 3.5)[..., None]
        twilight_upper_sky = color_space_navy[None, None, :] * v_frac[..., None] + np.array([25, 55, 105])[None, None, :] / 255.0 * (1.0 - v_frac[..., None])
        twilight_sky_matrix = np.clip(twilight_horiz_glow + twilight_upper_sky, 0, 1)
        
        # FIXED: Lowered the horizon baseline coefficients from [16, 26, 44] down to [11, 17, 30]
        # to ensure that 11:30 PM plates render a clean, rich midnight dark navy instead of a washed-out slate.
        night_sky_matrix = color_space_navy[None, None, :] * v_frac[..., None] + np.array([11, 17, 30]) / 255.0 * (1.0 - v_frac[..., None])
        
        # --- 4. MAP THE EARTH'S SHADOW WEDGE & THE BELT OF VENUS ---
        az_diff_rad = np.radians(X_mesh - sun_az_deg)
        h_shadow = np.degrees(np.arcsin(np.clip(np.sin(np.radians(sun_deg)) * np.cos(az_diff_rad), -1.0, 1.0)))
        
        # Calculate the soft pink anti-twilight backscattering strip immediately above the shadow
        belt_of_venus_mask = np.exp(-((Y_mesh - (h_shadow + 3.0)) / 3.0)**2) * np.clip((sun_deg + 5.0) / 5.0, 0, 1)
        belt_of_venus_mask = np.where(h_shadow < 0, belt_of_venus_mask, 0)
        
        # Add the pink scattering arch directly into the twilight matrix calculation
        twilight_sky_matrix[..., 0] += belt_of_venus_mask * 0.16  # Inject warm pink wavelengths
        twilight_sky_matrix[..., 1] += belt_of_venus_mask * 0.05
        twilight_sky_matrix[..., 2] += belt_of_venus_mask * 0.08
        
        # --- 5. EXECUTE CONTINUOUS ILLUMINATION COUPLING TRANSITION ---
        if sun_deg > 0:
            fade_day_to_twilight = np.clip(sun_deg / 6.0, 0, 1)
            bg_image = day_sky_matrix * fade_day_to_twilight + twilight_sky_matrix * (1.0 - fade_day_to_twilight)
        else:
            # FIXED: Extended the twilight fade out to -12.0 degrees so the deep 
            # indigo and faint horizon tones linger organically through nautical twilight.
            fade_twilight_to_night = np.clip((sun_deg + 12.0) / 12.0, 0, 1)
            bg_image = twilight_sky_matrix * fade_twilight_to_night + night_sky_matrix * (1.0 - fade_twilight_to_night)
            
        # Apply eye-adaptation tone mapping and exponential standard gamma encoding
        bg_image = np.clip(bg_image, 0.0, 1.0)
        
        # FIXED: Widened the gamma scaling window out to -12.0 degrees. This keeps 
        # the sky mid-tones beautifully lifted and blue at 9:45 PM, while still falling 
        # into a rich, deep broadcast black by 11:30 PM.
        gamma_exponent = 2.0 if sun_deg > 0 else np.clip(2.0 + (sun_deg / 12.0), 1.0, 2.0)
        bg_image = bg_image ** (1.0 / gamma_exponent)
        
        grid_color = "#ffffff" if sun_deg > 0 else ("#475569" if sun_deg > -6.0 else "#334155")

        # Draw the calculated continuous 2D analytical gradient mesh onto the background plane
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
                rgba_glow,
                extent=[az_min, az_max, 0, 40],
                origin="lower",
                aspect="auto",
                zorder=1,
                interpolation="bilinear"
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
                
                body_az = az.degrees
                body_alt = alt.degrees
                
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
            
            moon_az = m_az.degrees
            moon_alt = m_alt.degrees
            
            if direction == "North" and moon_az < 90:
                moon_az += 360
                
            if az_min <= moon_az <= az_max and 0 <= moon_alt <= 40:
                sun_body = eph['sun']
                m_pos = observer_loc.at(t).observe(moon_body).position.au
                s_pos = observer_loc.at(t).observe(sun_body).position.au
                
                m_dot_s = np.dot(m_pos, s_pos) / (np.linalg.norm(m_pos) * np.linalg.norm(s_pos))
                elongation = np.arccos(np.clip(m_dot_s, -1.0, 1.0))
                illuminated_fraction = 0.5 * (1.0 + np.cos(elongation))
                
                pabl_rad = np.arctan2(sun_alt.degrees - moon_alt, sun_az_deg - moon_az)
                
                r_x = 0.65  
                r_y = r_x * (12.0 / 90.0) / (6.75 / 40.0)
                
                num_points = 30
                phi = np.linspace(-np.pi/2, np.pi/2, num_points)
                
                x_outer_unit = np.cos(phi)
                y_outer_unit = np.sin(phi)
                
                phase_modifier = (illuminated_fraction - 0.5) * 2.0
                x_inner_unit = x_outer_unit * phase_modifier
                y_inner_unit = y_outer_unit
                
                cos_p, sin_p = np.cos(pabl_rad), np.sin(pabl_rad)
                
                verts = []
                for idx in range(num_points):
                    x_rot = x_outer_unit[idx] * cos_p - y_outer_unit[idx] * sin_p
                    y_rot = x_outer_unit[idx] * sin_p + y_outer_unit[idx] * cos_p
                    rx = x_rot * r_x
                    ry = y_rot * r_y
                    verts.append((moon_az + rx, moon_alt + ry))
                    
                for idx in reversed(range(num_points)):
                    x_rot = x_inner_unit[idx] * cos_p - y_inner_unit[idx] * sin_p
                    y_rot = x_inner_unit[idx] * sin_p + y_inner_unit[idx] * cos_p
                    rx = x_rot * r_x
                    ry = y_rot * r_y
                    verts.append((moon_az + rx, moon_alt + ry))
                    
                verts.append(verts[0]) 
                
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
                            size = max(1.5, (5.0 - mag) * 2.5)
                            ax.scatter(star_az, star_alt, s=size, color="#ffffff", alpha=0.55, zorder=20)
                            if show_labels:
                                ax.text(star_az + 0.4, star_alt + 0.4, name, color="#ffffff", fontsize=9, alpha=0.5, zorder=21)
                    except Exception:
                        continue

        # 6. FIXED SUBURBAN TREE HORIZON SILHOUETTE
        x_silhouette_space = np.linspace(az_min, az_max, 400)
        base_ground = 4.0 + 1.0 * np.sin(x_silhouette_space / 5)
        tree_canopy = 1.2 * np.sin(x_silhouette_space * 2.5) * np.cos(x_silhouette_space * 0.4)
        fine_foliage = 0.5 * np.sin(x_silhouette_space * 12.0)
        y_silhouette = base_ground + tree_canopy + fine_foliage
        y_silhouette = np.clip(y_silhouette, 2.0, 10.0)

        ax.fill_between(x_silhouette_space, -5, y_silhouette, color="#060c14", zorder=100)
        
        # Clean up gridline decorations and formatting
        ax.grid(True, color=grid_color, alpha=0.15, linestyle='--', zorder=2)
        
        # LOCKED FIXED 10-DEGREE SPACING WITH ALL LABELS FORCEFULLY HIDDEN
        ax.set_xticks(np.arange(az_min, az_max + 1, 10))
        ax.set_yticks(np.arange(0, 41, 10))
        
        # FORCE BOTH AXES TO HIDE ALL TICK TEXT AND NUMBER LABELS NATIVELY
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Strip out the spine borders to collapse bounding margins edge-to-edge
        for spine in ax.spines.values():
            spine.set_visible(False)

        # --- DISPLAY & DOWNLOAD ---
        st.pyplot(fig)
        
        img_buf = io.BytesIO()
        fig.savefig(
            img_buf, 
            format="png", 
            dpi=150, 
            facecolor="none",  
            edgecolor="none",
            pad_inches=0.0
        )
        img_buf.seek(0)
        
        st.download_button(
            label="💾 Download High-Res PNG for Editing / On-Air",
            data=img_buf,
            file_name=f"custom_sky_{direction}.png",
            mime="image/png"
        )

        # Completely clear out the figure state to prevent scaling bleed on re-runs
        plt.close(fig)

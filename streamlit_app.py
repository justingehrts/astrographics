import streamlit as st
from datetime import datetime, time
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import io
import numpy as np
from skyfield.api import load, wgs84, Star

st.set_page_config(layout="wide", page_title="Custom Sky Graphic Generator")
st.title("🌌 Broadcast Sky Graphic Generator")
st.write("A lightweight, reliable engine rendering clean astronomical plates with native horizon silhouettes.")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.header("1. Observation Settings")
obs_date = st.sidebar.date_input("Select Date", datetime.now().date())

time_options, time_labels = [], []
for hour in range(24):
    for minute in [0, 15, 30, 45]:
        t_obj = time(hour, minute)
        time_options.append(t_obj)
        time_labels.append(t_obj.strftime("%I:%M %p"))

selected_time_index = time_labels.index("09:15 PM") if "09:15 PM" in time_labels else 0
obs_time = st.sidebar.selectbox(
    "Select Time", options=time_options,
    format_func=lambda x: x.strftime("%I:%M %p"), index=selected_time_index
)

tz_options = ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]
selected_tz = st.sidebar.selectbox("Time Zone", tz_options, index=0)

lat = st.sidebar.number_input("Latitude",  value=40.00, step=0.01, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=-83.10, step=0.01, format="%.2f")

st.sidebar.header("2. View Window")
direction = st.sidebar.selectbox(
    "Looking Direction",
    ["North","Northeast","East","Southeast","South","Southwest","West","Northwest"],
    index=6
)

st.sidebar.header("3. Graphic Toggles")
show_labels    = st.sidebar.checkbox("Show Object Labels", value=True)
star_brightness = st.sidebar.slider("Star Visibility Limit", 1.0, 4.5, 2.5, step=0.5)

# Turbidity: the single most important knob for twilight realism
turbidity = st.sidebar.slider(
    "Atmosphere Turbidity", 1.5, 6.0, 2.5, step=0.25,
    help="1.5 = crystal clear mountain air (vivid purple/blue twilight). "
         "2.5 = typical suburban. 4–6 = hazy/humid summer (deep orange/red twilight)."
)

sky_conditions = {
    1.0: "🏙️ Heavy City Light Pollution",
    1.5: "🌆 Urban Sky",
    2.0: "🏘️ Bright Suburban Sky",
    2.5: "🏡 Typical Suburban Sky",
    3.0: "🌳 Dark Suburban / Rural Fringe",
    3.5: "🚜 Rural Country Sky",
    4.0: "🌌 Very Dark Sky",
    4.5: "✨ Pristine Dark Sky",
}
st.sidebar.caption(f"**Viewport:** {sky_conditions[star_brightness]}")

az_map = {
    "North": (315, 405), "Northeast": (0, 90),  "East": (45, 135),
    "Southeast": (90, 180), "South": (135, 225), "Southwest": (180, 270),
    "West": (225, 315), "Northwest": (270, 360),
}
az_min, az_max = az_map[direction]


# ── PREETHAM xyY SKY MODEL ────────────────────────────────────────────────────
# Direct Python port of Stellarium's Skylight.cpp (Preetham, Shirley & Smits 1999)
# "A Practical Analytic Model for Daylight", SIGGRAPH '99

def preetham_perez(theta, gamma, zenith_val, A, B, C, D, E):
    """
    The Perez all-weather sky distribution function.
    theta = angle from zenith to sky point (radians)
    gamma = angle between sky point and sun (radians)
    """
    cos_theta = np.cos(theta)
    cos_gamma = np.cos(gamma)
    # Avoid divide-by-zero at zenith
    cos_theta_safe = np.where(np.abs(cos_theta) < 1e-6, 1e-6, cos_theta)
    num   = (1 + A * np.exp(B / cos_theta_safe)) * (1 + C * np.exp(D * gamma) + E * cos_gamma**2)
    denom = (1 + A * np.exp(B)) * (1 + C * np.exp(D * np.arccos(np.clip(cos_theta, -1, 1))) + E)
    # denom is a scalar per-pixel; we compute it for the zenith point (theta=0, gamma=sun_theta)
    return num


def preetham_coeffs(T):
    """
    Return Perez distribution coefficients for x, y chromaticity and Y luminance
    as functions of turbidity T. Tables from Preetham 1999, Section A.1.
    """
    Ax = -0.0193*T - 0.2592;  Bx = -0.0665*T + 0.0008;  Cx = -0.0004*T + 0.2125
    Dx = -0.0641*T - 0.8989;  Ex =  0.0886*T + 0.0452

    Ay = -0.0167*T - 0.2608;  By = -0.0950*T + 0.0092;  Cy = -0.0079*T + 0.2102
    Dy = -0.0441*T - 1.6537;  Ey =  0.0299*T + 0.0529

    AY = -0.1787*T - 1.4630;  BY = -0.3554*T + 0.4275;  CY =  0.1198*T + 5.3251
    DY = -0.0029*T - 2.5771;  EY =  0.0516*T + 0.3703

    return (Ax,Bx,Cx,Dx,Ex), (Ay,By,Cy,Dy,Ey), (AY,BY,CY,DY,EY)


def zenith_xyY(T, sun_theta):
    """
    Zenith chromaticity (x,y) and luminance (Y in kcd/m²) from turbidity and
    solar zenith angle. Preetham 1999, Appendix A.
    """
    t2 = T * T
    th  = sun_theta
    th2 = th * th
    th3 = th2 * th

    xz = ( 0.00166*th3 - 0.00375*th2 + 0.00209*th) * t2 \
       + (-0.02903*th3 + 0.06377*th2 - 0.03202*th + 0.00394) * T \
       + ( 0.11693*th3 - 0.21196*th2 + 0.06052*th + 0.25886)

    yz = ( 0.00275*th3 - 0.00610*th2 + 0.00317*th) * t2 \
       + (-0.04214*th3 + 0.08970*th2 - 0.04153*th + 0.00516) * T \
       + ( 0.15346*th3 - 0.26756*th2 + 0.06670*th + 0.26688)

    # Zenith luminance (kcd/m²)
    chi = (4.0/9.0 - T/120.0) * (np.pi - 2*sun_theta)
    Yz  = (4.0453*T - 4.9710) * np.tan(chi) - 0.2155*T + 2.4192

    return xz, yz, max(Yz, 0.0)


def xyY_to_RGB(x, y, Y):
    """
    CIE xyY → XYZ → linear sRGB, then gamma-encode and clip to [0,1].
    Handles arrays or scalars.
    """
    # Guard against Y<=0 or bad chromaticity
    Y = np.maximum(Y, 0.0)
    y_safe = np.where(np.abs(y) < 1e-6, 1e-6, y)

    X = (Y / y_safe) * x
    Z = (Y / y_safe) * (1.0 - x - y)

    # sRGB D65 matrix (IEC 61966-2-1)
    R =  3.2406*X - 1.5372*Y - 0.4986*Z
    G = -0.9689*X + 1.8758*Y + 0.0415*Z
    B =  0.0557*X - 0.2040*Y + 1.0570*Z

    # Clamp negatives, then gamma encode
    R = np.maximum(R, 0.0)
    G = np.maximum(G, 0.0)
    B = np.maximum(B, 0.0)

    def gamma(c):
        return np.where(c <= 0.0031308, 12.92*c, 1.055*c**(1/2.4) - 0.055)

    return np.clip(gamma(R), 0, 1), np.clip(gamma(G), 0, 1), np.clip(gamma(B), 0, 1)


def build_sky_image(sun_az_deg, sun_alt_deg, turbidity, az_min, az_max,
                    x_pixels=150, y_pixels=100, alt_max=40.0):
    """
    Render a 2-D sky image using the Preetham analytic daylight model.
    Returns an (y_pixels, x_pixels, 3) float32 RGB array clipped to [0,1].

    Twilight & night are handled by:
      - Smoothly scaling luminance to zero below the horizon
      - Blending in a physically-derived night sky colour for sun_alt < -6°
      - Preserving the anti-twilight arch (Earth shadow opposite the sun)
    """
    T = turbidity
    sun_az_r  = np.radians(sun_az_deg)
    sun_alt_r = np.radians(sun_alt_deg)
    sun_zen_r = np.pi/2 - sun_alt_r          # zenith angle of sun

    coeffs_x, coeffs_y, coeffs_Y = preetham_coeffs(T)
    xz, yz, Yz = zenith_xyY(T, max(sun_zen_r, 0.0))  # clamp sun below horizon

    # Build mesh
    x_sp = np.linspace(az_min, az_max, x_pixels)
    y_sp = np.linspace(0, alt_max, y_pixels)
    AZ, ALT = np.meshgrid(x_sp, y_sp)

    az_r  = np.radians(AZ)
    alt_r = np.radians(ALT)
    zen_r = np.pi/2 - alt_r   # zenith angle of each sky pixel

    # Angle between sky pixel and sun (gamma)
    cos_gamma = (np.sin(alt_r)*np.sin(sun_alt_r)
                 + np.cos(alt_r)*np.cos(sun_alt_r)*np.cos(az_r - sun_az_r))
    gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))

    # Perez function values at each pixel and at zenith (theta=0, gamma=sun_zen)
    def F(theta, gamma, A, B, C, D, E):
        cos_t  = np.cos(theta)
        cos_t  = np.where(np.abs(cos_t) < 1e-6, 1e-6, cos_t)
        return ((1 + A*np.exp(B/cos_t)) * (1 + C*np.exp(D*gamma) + E*np.cos(gamma)**2))

    def F0(gamma0, A, B, C, D, E):
        # At zenith theta=0 → cos=1
        return ((1 + A*np.exp(B)) * (1 + C*np.exp(D*gamma0) + E*np.cos(gamma0)**2))

    Fx  = F(zen_r, gamma, *coeffs_x)
    Fy  = F(zen_r, gamma, *coeffs_y)
    FY  = F(zen_r, gamma, *coeffs_Y)

    sun_gamma_at_z = sun_zen_r   # angle from zenith to sun = sun's own zenith angle
    F0x = F0(sun_gamma_at_z, *coeffs_x)
    F0y = F0(sun_gamma_at_z, *coeffs_y)
    F0Y = F0(sun_gamma_at_z, *coeffs_Y)

    x_sky = xz * (Fx / np.where(F0x < 1e-9, 1e-9, F0x))
    y_sky = yz * (Fy / np.where(F0y < 1e-9, 1e-9, F0y))
    Y_sky = Yz * (FY / np.where(F0Y < 1e-9, 1e-9, F0Y))

    # ── Twilight / Night extensions ──────────────────────────────────────────
    # When the sun is below the horizon Preetham breaks down (it was designed for
    # sun_alt ≥ 0).  We use three physically-motivated corrections:

    # 1. Luminance decay: scale Y by how much of the atmosphere is still lit.
    #    Civil twilight (−6°) ≈ 3.3 cd/m², nautical (−12°) ≈ 0.01, astro (−18°) ≈ 0.
    #    We model this with an exponential ramp on sun altitude.
    if sun_alt_deg <= 0:
        dip = abs(sun_alt_deg)          # 0 → 18
        # Luminance falls ~4 orders of magnitude from sunset to astro twilight
        lum_scale = np.exp(-dip * 0.38)     # at -6° → ×0.10,  -12° → ×0.01
        Y_sky = Y_sky * lum_scale

    # 2. Night-sky base: add a faint dark-blue background luminance floor
    #    (zodiacal + airglow level, ~0.0002 kcd/m²)
    night_floor_Y = 0.0002
    night_floor_x = 0.310   # slightly blue-shifted relative to D65 (0.3127)
    night_floor_y = 0.320

    # 3. Anti-twilight arch: at civil/nautical twilight the Belt of Venus appears
    #    as a pinkish-purple band on the anti-solar horizon.  It fades by −6°.
    #    We inject it as a gentle red+blue boost on the horizon opposite the sun.
    atw_strength = 0.0
    atw_rgb = np.zeros((y_pixels, x_pixels, 3))
    if -8.0 < sun_alt_deg <= 0:
        atw_strength = np.clip((sun_alt_deg + 8.0) / 8.0, 0, 1)  # 0 at -8°, 1 at 0°
        # Anti-solar direction
        anti_az_r = sun_az_r + np.pi
        # Horizontal distance to anti-solar point
        d_anti = np.cos(az_r - anti_az_r)   # 1 = directly opposite sun
        alt_frac = np.clip(1.0 - ALT / 12.0, 0, 1)   # only low sky, ≤12°
        atw_mask = np.clip(d_anti, 0, 1) * alt_frac
        # Pink-purple colour (Belt of Venus / Earth shadow gradient)
        atw_r = atw_mask * 0.40 * atw_strength
        atw_g = atw_mask * 0.10 * atw_strength
        atw_b = atw_mask * 0.35 * atw_strength
        atw_rgb[..., 0] = atw_r
        atw_rgb[..., 1] = atw_g
        atw_rgb[..., 2] = atw_b

    # ── Tone mapping ─────────────────────────────────────────────────────────
    # Preetham Y is in kcd/m².  Map to display range with a simple Reinhard-style
    # exposure so that a noon sky (~10 kcd/m²) → white, and twilight is dark.
    # You can tweak EXPOSURE to taste — lower = brighter overall.
    EXPOSURE = 8.0          # kcd/m² that maps to display white
    Y_display = Y_sky / EXPOSURE

    R, G, B = xyY_to_RGB(x_sky, y_sky, Y_display)

    # Blend night floor in proportion to how dark the sky has gotten
    night_blend = np.clip(1.0 - np.max(np.stack([R, G, B], axis=-1), axis=-1) / 0.06, 0, 1)
    nR, nG, nB = xyY_to_RGB(
        np.full_like(R, night_floor_x),
        np.full_like(R, night_floor_y),
        np.full_like(R, night_floor_Y / EXPOSURE)
    )
    R = R*(1-night_blend) + nR*night_blend
    G = G*(1-night_blend) + nG*night_blend
    B = B*(1-night_blend) + nB*night_blend

    # Composite anti-twilight arch
    R = np.clip(R + atw_rgb[..., 0], 0, 1)
    G = np.clip(G + atw_rgb[..., 1], 0, 1)
    B = np.clip(B + atw_rgb[..., 2], 0, 1)

    bg_image = np.stack([R, G, B], axis=-1).astype(np.float32)
    return bg_image


# ── GENERATE ─────────────────────────────────────────────────────────────────
if st.button("Generate Sky Graphic", type="primary"):
    with st.spinner("Computing high-fidelity Preetham atmospheric sky model..."):

        dt_local = datetime.combine(obs_date, obs_time, tzinfo=ZoneInfo(selected_tz))
        dt_utc   = dt_local.astimezone(ZoneInfo("UTC"))

        ts   = load.timescale()
        t    = ts.from_datetime(dt_utc)
        eph  = load('de421.bsp')
        earth = eph['earth']
        observer_loc = earth + wgs84.latlon(lat, lon)

        sun = eph['sun']
        sun_astrometric = observer_loc.at(t).observe(sun)
        sun_alt, sun_az, _ = sun_astrometric.apparent().altaz()
        sun_deg    = sun_alt.degrees
        sun_az_deg = sun_az.degrees

        # ── SKY GRADIENT ──────────────────────────────────────────────────────
        bg_image = build_sky_image(sun_az_deg, sun_deg, turbidity, az_min, az_max)

        fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100, facecolor='none')
        ax.set_xlim(az_min, az_max)
        ax.set_ylim(0, 40)

        ax.imshow(
            bg_image,
            extent=[az_min, az_max, 0, 40],
            origin="lower", aspect="auto", zorder=0
        )

        # ── CITY GLOW ─────────────────────────────────────────────────────────
        if star_brightness <= 1.5 and sun_deg <= 0:
            import matplotlib
            x_glow, y_glow = 200, 100
            x_g  = np.linspace(az_min, az_max, x_glow)
            y_g  = np.linspace(0, 40, y_glow)
            Xg, Yg = np.meshgrid(x_g, y_g)
            caz   = (az_min + az_max) / 2.0
            glow  = np.exp(-((Xg - caz)/70.0)**2 - (Yg/10.0)**2)
            rgba_glow = np.zeros((y_glow, x_glow, 4))
            rgba_glow[..., :3] = matplotlib.colors.to_rgb(
                "#fbf8f0" if star_brightness == 1.0 else "#f4efe2"
            )
            rgba_glow[..., 3] = glow * 0.28
            ax.imshow(rgba_glow, extent=[az_min, az_max, 0, 40],
                      origin="lower", aspect="auto", zorder=1, interpolation="bilinear")

        # ── PLANETS ───────────────────────────────────────────────────────────
        bodies = {
            'mercury': (eph['mercury'], 16, 'Mercury'),
            'venus':   (eph['venus'],   35, 'Venus'),
            'mars':    (eph['mars'],    16, 'Mars'),
            'jupiter': (eph['jupiter_barycenter'], 45, 'Jupiter'),
            'saturn':  (eph['saturn_barycenter'],  24, 'Saturn'),
        }
        for name, (body, size, label) in bodies.items():
            try:
                ast  = observer_loc.at(t).observe(body)
                alt, az, _ = ast.apparent().altaz()
                baz, balt = az.degrees, alt.degrees
                if direction == "North" and baz < 90:
                    baz += 360
                if az_min <= baz <= az_max and 0 <= balt <= 40:
                    ax.scatter(baz, balt, s=size, color="#ffffff", zorder=50)
                    if show_labels:
                        ax.text(baz+0.5, balt+0.5, label, color="#ffffff",
                                fontsize=10, weight='bold', zorder=51)
            except Exception:
                continue

        # ── MOON ──────────────────────────────────────────────────────────────
        try:
            moon_body = eph['moon']
            m_ast = observer_loc.at(t).observe(moon_body)
            m_alt, m_az, _ = m_ast.apparent().altaz()
            moon_az, moon_alt = m_az.degrees, m_alt.degrees
            if direction == "North" and moon_az < 90:
                moon_az += 360
            if az_min <= moon_az <= az_max and 0 <= moon_alt <= 40:
                m_pos = observer_loc.at(t).observe(moon_body).position.au
                s_pos = observer_loc.at(t).observe(eph['sun']).position.au
                m_dot_s = np.dot(m_pos, s_pos) / (np.linalg.norm(m_pos)*np.linalg.norm(s_pos))
                elongation = np.arccos(np.clip(m_dot_s, -1.0, 1.0))
                illum = 0.5*(1.0 + np.cos(elongation))
                pabl_rad = np.arctan2(sun_deg - moon_alt, sun_az_deg - moon_az)

                r_x = 0.65
                r_y = r_x * (12.0/90.0) / (6.75/40.0)
                num_pts = 30
                phi = np.linspace(-np.pi/2, np.pi/2, num_pts)
                xo, yo = np.cos(phi), np.sin(phi)
                pm = (illum - 0.5)*2.0
                xi, yi = xo*pm, yo
                cp, sp = np.cos(pabl_rad), np.sin(pabl_rad)
                verts = []
                for i in range(num_pts):
                    xr = xo[i]*cp - yo[i]*sp;  yr = xo[i]*sp + yo[i]*cp
                    verts.append((moon_az + xr*r_x, moon_alt + yr*r_y))
                for i in reversed(range(num_pts)):
                    xr = xi[i]*cp - yi[i]*sp;  yr = xi[i]*sp + yi[i]*cp
                    verts.append((moon_az + xr*r_x, moon_alt + yr*r_y))
                verts.append(verts[0])
                codes = [Path.MOVETO]+[Path.LINETO]*(len(verts)-2)+[Path.CLOSEPOLY]
                ax.add_patch(patches.PathPatch(Path(verts, codes),
                             facecolor='#ffffff', edgecolor='none', zorder=50))
                if illum < 0.90:
                    ax.add_patch(patches.Ellipse((moon_az, moon_alt), r_x*2, r_y*2,
                                 facecolor='#ffffff', alpha=0.08, edgecolor='none', zorder=49))
                if show_labels:
                    ax.text(moon_az+r_x+0.4, moon_alt+0.6, "Moon",
                            color="#ffffff", fontsize=11, weight='bold', zorder=51)
        except Exception:
            pass

        # ── STARS ─────────────────────────────────────────────────────────────
        if sun_deg <= -6:
            star_data = [
                ("Polaris",   1.97, (2,31,49.1),  (89,15,51)),
                ("Vega",      0.03, (18,36,56.3),  (38,47,1)),
                ("Capella",   0.08, (5,16,41.4),   (45,59,53)),
                ("Arcturus", -0.05, (14,15,39.7),  (19,10,57)),
                ("Betelgeuse",0.50, (5,55,10.3),   (7,24,25)),
                ("Procyon",   0.34, (7,39,18.1),   (5,13,30)),
                ("Pollux",    1.14, (7,45,18.9),   (28,1,34)),
                ("Castor",    1.58, (7,34,36.0),   (31,53,18)),
                ("Spica",     0.98, (13,25,11.6),  (-11,9,41)),
                ("Altair",    0.76, (19,50,47.0),  (8,52,6)),
                ("Deneb",     1.25, (20,41,25.9),  (45,16,49)),
                ("Regulus",   1.36, (10,8,22.3),   (11,58,2)),
            ]
            for name, mag, ra, dec in star_data:
                if mag <= star_brightness:
                    try:
                        s_ast = observer_loc.at(t).observe(Star(ra_hours=ra, dec_degrees=dec))
                        s_alt, s_az, _ = s_ast.apparent().altaz()
                        saz, salt = s_az.degrees, s_alt.degrees
                        if direction == "North" and saz < 90:
                            saz += 360
                        if az_min <= saz <= az_max and 0 <= salt <= 40:
                            size = max(1.5, (5.0 - mag)*2.5)
                            ax.scatter(saz, salt, s=size, color="#ffffff", alpha=0.55, zorder=20)
                            if show_labels:
                                ax.text(saz+0.4, salt+0.4, name, color="#ffffff",
                                        fontsize=9, alpha=0.5, zorder=21)
                    except Exception:
                        continue

        # ── HORIZON SILHOUETTE ────────────────────────────────────────────────
        x_sil = np.linspace(az_min, az_max, 400)
        base  = 4.0 + 1.0*np.sin(x_sil/5)
        canopy= 1.2*np.sin(x_sil*2.5)*np.cos(x_sil*0.4)
        fine  = 0.5*np.sin(x_sil*12.0)
        y_sil = np.clip(base + canopy + fine, 2.0, 10.0)
        ax.fill_between(x_sil, -5, y_sil, color="#060c14", zorder=100)

        # ── GRID & LABELS ─────────────────────────────────────────────────────
        grid_color = "#ffffff" if sun_deg > 0 else ("#475569" if sun_deg > -6 else "#334155")
        ax.grid(True, color=grid_color, alpha=0.15, linestyle='--', zorder=2)
        ax.set_yticks(np.arange(0, 41, 10))
        ax.set_yticklabels([f"{int(y)}°" for y in np.arange(0, 41, 10)])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ── OUTPUT ────────────────────────────────────────────────────────────
        st.pyplot(fig)

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=150,
                    facecolor="none", edgecolor="none", pad_inches=0.0)
        img_buf.seek(0)
        st.download_button(
            label="💾 Download High-Res PNG",
            data=img_buf,
            file_name=f"sky_{direction}.png",
            mime="image/png"
        )
        plt.close(fig)

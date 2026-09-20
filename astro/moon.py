"""Moon appearance: correct bright-limb orientation and real angular size.

Replaces two approximations in the original renderer:

- The bright-limb (terminator) orientation was computed as a flat alt/az
  difference (`arctan2(sun_alt - moon_alt, sun_az - moon_az)`), a
  plane approximation that's least accurate near the horizon -- exactly
  where this app operates (0-40deg altitude). This module uses the
  standard spherical formula instead (Meeus, *Astronomical Algorithms*,
  ch. 48-49): the bright limb's position angle from equatorial North,
  corrected to the local horizon frame by the parallactic angle.
- The Moon's angular size was a fixed constant; it actually varies about
  +/-12% between perigee and apogee, so this module derives it from the
  real Earth-Moon distance for the given observation.
"""
import numpy as np

MOON_RADIUS_KM = 1737.4


def angular_radius_deg(distance_km):
    """Moon's true angular radius (semi-diameter) at the given distance."""
    return np.degrees(np.arcsin(MOON_RADIUS_KM / distance_km))


def bright_limb_position_angle_deg(sun_ra_hours, sun_dec_deg, moon_ra_hours, moon_dec_deg):
    """Position angle of the Moon's bright limb (direction toward the Sun
    as seen from the Moon's center), measured from equatorial North,
    turning east. Meeus eq. 48.5."""
    ra_s, dec_s = np.radians(sun_ra_hours * 15.0), np.radians(sun_dec_deg)
    ra_m, dec_m = np.radians(moon_ra_hours * 15.0), np.radians(moon_dec_deg)
    d_ra = ra_s - ra_m
    y = np.cos(dec_s) * np.sin(d_ra)
    x = np.sin(dec_s) * np.cos(dec_m) - np.cos(dec_s) * np.sin(dec_m) * np.cos(d_ra)
    return np.degrees(np.arctan2(y, x))


def parallactic_angle_deg(latitude_deg, dec_deg, hour_angle_deg):
    """Parallactic angle: the angle at the Moon between the direction to
    the zenith and the direction to the celestial pole. Needed to rotate
    the bright-limb position angle from the equatorial frame (measured
    from North) into the local horizon frame (measured from the zenith)."""
    phi, dec, ha = np.radians(latitude_deg), np.radians(dec_deg), np.radians(hour_angle_deg)
    return np.degrees(np.arctan2(
        np.sin(ha),
        np.tan(phi) * np.cos(dec) - np.sin(dec) * np.cos(ha),
    ))


def bright_limb_plot_angle_rad(position_angle_deg, parallactic_deg):
    """Converts the bright-limb position angle (from equatorial North,
    turning east) into the rotation angle used by the local alt-az plot,
    where 0 rad points toward increasing azimuth (+x) and pi/2 rad points
    toward the zenith (+y, straight up in the graphic)."""
    vertex_angle_deg = position_angle_deg - parallactic_deg  # from zenith, turning east
    return np.radians(90.0 - vertex_angle_deg)

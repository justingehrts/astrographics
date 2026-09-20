"""Moon appearance: correct bright-limb orientation and real angular size.

Replaces two approximations in the original renderer:

- The bright-limb (terminator) orientation was computed as a flat alt/az
  difference (`arctan2(sun_alt - moon_alt, sun_az - moon_az)`), a plane
  approximation that's least accurate near the horizon -- exactly where
  this app operates (0-40deg altitude). This module computes it directly
  in the horizontal (alt/az) frame instead, using the same spherical
  position-angle formula astronomers use for the equatorial (RA/Dec)
  bright limb (Meeus, *Astronomical Algorithms*, ch. 48), applied to
  alt/az in place of Dec/RA. That's algebraically the same quantity the
  standard equatorial-PA-plus-parallactic-angle method produces (both are
  "position angle of the Sun as seen from the Moon, relative to local
  up"), just computed in one step directly in the frame the plot already
  uses, with no separate parallactic-angle correction, hour angle, or
  sidereal time needed -- and verified numerically (not just re-derived
  by hand) against the geometric invariant that the bright limb must
  point from the Moon roughly toward the Sun's plotted direction.
- The Moon's angular size was a fixed constant; it actually varies about
  +/-12% between perigee and apogee, so this module derives it from the
  real Earth-Moon distance for the given observation.
"""
import numpy as np

MOON_RADIUS_KM = 1737.4


def angular_radius_deg(distance_km):
    """Moon's true angular radius (semi-diameter) at the given distance."""
    return np.degrees(np.arcsin(MOON_RADIUS_KM / distance_km))


def bright_limb_plot_angle_rad(sun_az_deg, sun_alt_deg, moon_az_deg, moon_alt_deg):
    """Rotation angle for the bright-limb crescent shape in the local
    alt-az plot: 0 rad points toward increasing azimuth (+x) and pi/2 rad
    points toward the zenith (+y, straight up in the graphic)."""
    d_az = np.radians(sun_az_deg - moon_az_deg)
    alt_s, alt_m = np.radians(sun_alt_deg), np.radians(moon_alt_deg)
    y = np.cos(alt_s) * np.sin(d_az)
    x = np.sin(alt_s) * np.cos(alt_m) - np.cos(alt_s) * np.sin(alt_m) * np.cos(d_az)
    position_angle_deg = np.degrees(np.arctan2(y, x))
    return np.radians(90.0 - position_angle_deg)

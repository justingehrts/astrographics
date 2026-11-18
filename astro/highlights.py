"""Highlights for a chosen date: which planets and the Moon are worth showing, when
they are best placed, and where to point the camera.

An object is listed when, at some time that day (15-minute steps, local time):
  - it is at least MIN_ALTITUDE_DEG above the horizon, and
  - the Sun is at least MAX_SUN_ALTITUDE_DEG below the horizon (so no daytime Venus), and
  - for a planet, it is clearly visible by the app's own twilight/extinction model
    (the same opacity rule that draws it).
Each object's best time is when it stands highest among its visible times.
"""
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from skyfield import almanac
from skyfield.api import wgs84
from skyfield.magnitudelib import planetary_magnitude

from astro import extinction, stars

MIN_ALTITUDE_DEG = 10.0
MAX_SUN_ALTITUDE_DEG = -3.0   # sky has started to darken: after sunset / before sunrise
MIN_PLANET_ALPHA = 0.5   # "clearly visible": at least half the opacity of a full-strength point
STEP_MINUTES = 15

PLANETS = (
    ("Mercury", "mercury"), ("Venus", "venus"), ("Mars", "mars"),
    ("Jupiter", "jupiter_barycenter"), ("Saturn", "saturn_barycenter"),
)
_COMPASS_16 = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
               "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


@dataclass(frozen=True)
class Highlight:
    name: str          # "Moon", "Saturn"
    detail: str        # "62% lit, waxing" for the Moon, "" for planets
    when: time         # best time that day, on the 15-minute grid
    az_deg: float
    alt_deg: float
    magnitude: float   # for ordering planets, brightest first (the Moon sorts first)

    @property
    def direction(self):
        return compass_name(self.az_deg)


def compass_name(az_deg):
    """16-point compass name for an azimuth."""
    return _COMPASS_16[int(((az_deg % 360.0) + 11.25) // 22.5) % 16]


def _local_grid(day, zone_name):
    zone = ZoneInfo(zone_name)
    base = datetime.combine(day, time(0, 0))
    steps = 24 * 60 // STEP_MINUTES
    return [(base + timedelta(minutes=STEP_MINUTES * i)) for i in range(steps)], zone


def find_highlights(ts, eph, lat, lon, day, zone_name, turbidity,
                    temperature_c=10.0, pressure_mbar=1010.0):
    """Highlights for `day` at (lat, lon), Moon first then planets brightest first."""
    naive_times, zone = _local_grid(day, zone_name)
    utc = [t.replace(tzinfo=zone).astimezone(ZoneInfo("UTC")) for t in naive_times]
    t = ts.from_datetimes(utc)
    observer = eph["earth"] + wgs84.latlon(lat, lon)
    at = observer.at(t)

    def altaz(body):
        astrometric = at.observe(body)
        alt, az, _ = astrometric.apparent().altaz(temperature_C=temperature_c, pressure_mbar=pressure_mbar)
        return astrometric, alt.degrees, az.degrees

    _, sun_alt, _ = altaz(eph["sun"])
    twilight_limit = np.array([stars.twilight_limiting_magnitude(a) for a in sun_alt])
    found = []

    dark_enough = sun_alt <= MAX_SUN_ALTITUDE_DEG

    # Moon: any time the sky is dark enough and the Moon is high enough.
    _, moon_alt, moon_az = altaz(eph["moon"])
    ok = (moon_alt >= MIN_ALTITUDE_DEG) & dark_enough
    if ok.any():
        i = int(np.argmax(np.where(ok, moon_alt, -np.inf)))
        phase = float(almanac.moon_phase(eph, t[i]).degrees)  # 0 new, 90 first quarter, 180 full
        lit = 0.5 * (1.0 - np.cos(np.radians(phase)))
        found.append(Highlight(
            "Moon", f"{lit * 100:.0f}% lit, {'waxing' if phase < 180.0 else 'waning'}",
            naive_times[i].time(), float(moon_az[i]), float(moon_alt[i]), -99.0))

    for label, key in PLANETS:
        try:
            astrometric, alt, az = altaz(eph[key])
            mag = np.asarray(planetary_magnitude(astrometric), dtype=float)
        except Exception:
            continue
        mag = np.where(np.isnan(mag), 0.0, mag)  # outside the magnitude model's phase range
        eff = mag + extinction.magnitude_loss(alt, turbidity)
        alpha = stars.visibility_alpha(eff, twilight_limit)
        ok = (alt >= MIN_ALTITUDE_DEG) & (alpha >= MIN_PLANET_ALPHA) & dark_enough
        if ok.any():
            i = int(np.argmax(np.where(ok, alt, -np.inf)))
            found.append(Highlight(label, "", naive_times[i].time(), float(az[i]), float(alt[i]), float(mag[i])))

    return sorted(found, key=lambda h: h.magnitude)

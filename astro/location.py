"""Location-aware defaults for the turbidity / star-brightness sliders.

There's no free, keyless light-pollution atlas API (the well-known
light-pollution maps are interactive websites, not documented endpoints),
so this uses a heuristic instead: reverse-geocode the observer's
coordinates via OpenStreetMap's free, keyless Nominatim API and classify
urban/suburban/rural from the returned place type (and population, when
available), then map that to a starting point for the sliders. This is
explicitly a starting point, not a measurement -- the sliders stay fully
user-adjustable, and this fails soft (falls back to a fixed default)
rather than block the UI if the lookup is slow or unavailable.

Nominatim's usage policy asks for a descriptive User-Agent and at most
~1 request/second, both fine for this app's per-click usage.
"""
import logging

import requests

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
REQUEST_TIMEOUT_S = 5
USER_AGENT = "astrographics-sky-graphic-generator (personal broadcast tool)"

# (turbidity, star_brightness) starting points, matching the sky_conditions
# captions already defined in streamlit_app.py.
URBAN_DEFAULT = (2.5, 1.5)
SUBURBAN_DEFAULT = (2.0, 2.5)
RURAL_DEFAULT = (1.5, 3.5)
FALLBACK_DEFAULT = SUBURBAN_DEFAULT

_URBAN_PLACE_TYPES = {"city", "town"}
_SUBURBAN_PLACE_TYPES = {"suburb", "village", "borough", "quarter", "neighbourhood", "municipality"}
_LARGE_CITY_POPULATION = 200_000


def suggest_sky_defaults(lat, lon):
    """Returns a (turbidity, star_brightness) starting point for the given
    location. Never raises."""
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 12},
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        address = data.get("address", {})
        place_type = data.get("addresstype") or data.get("type") or ""
        population = int(address.get("population", 0) or 0)

        if population >= _LARGE_CITY_POPULATION or place_type in _URBAN_PLACE_TYPES:
            return URBAN_DEFAULT
        if place_type in _SUBURBAN_PLACE_TYPES:
            return SUBURBAN_DEFAULT
        if place_type:
            return RURAL_DEFAULT
        return FALLBACK_DEFAULT
    except Exception:
        logger.exception("Reverse geocoding for sky-condition defaults failed")
        return FALLBACK_DEFAULT

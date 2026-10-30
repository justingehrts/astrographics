"""City/state -> coordinates (and time zone) lookup for the location box.

Primary source is Open-Meteo's free, keyless geocoding API (the app already
uses Open-Meteo for forecasts); it returns the state, coordinates, population
and IANA time zone. If it is unreachable or finds nothing, OpenStreetMap's
Nominatim search (already used for sky-condition defaults, with the same
descriptive User-Agent) is tried; it has no time zone, so that field is None.
Everything fails soft: errors become an empty result list.
"""
import logging
import re

import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
REQUEST_TIMEOUT_S = 6
USER_AGENT = "astrographics-sky-graphic-generator (personal broadcast tool)"

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
_STATE_BY_NAME = {name.lower(): name for name in US_STATES.values()}


def _state_name(token):
    """Full state name for an abbreviation ("oh") or a name ("ohio"), else None."""
    token = token.strip().rstrip(".").lower()
    if token.upper() in US_STATES:
        return US_STATES[token.upper()]
    return _STATE_BY_NAME.get(token)


def parse_query(text):
    """(city, state_name_or_None) from "Columbus, OH", "Columbus, Ohio", "Columbus OH"
    or just "Columbus". A trailing "USA"/"US" is ignored."""
    text = re.sub(r"[,\s]+(usa|us|united states)\s*$", "", (text or "").strip(), flags=re.I).strip(" ,")
    if not text:
        return "", None
    if "," in text:
        city, _, tail = text.rpartition(",")
        state = _state_name(tail)
        if state:
            return city.strip(), state
        return " ".join(text.replace(",", " ").split()), None
    words = text.split()
    for n in (3, 2, 1):  # longest state name first ("North Carolina" before "Carolina")
        if len(words) > n and (state := _state_name(" ".join(words[-n:]))):
            return " ".join(words[:-n]), state
    return text, None


def _label(name, state, country_code):
    parts = [name]
    if state:
        parts.append(state)
    if country_code and country_code.upper() != "US":
        parts.append(country_code.upper())
    return ", ".join(parts)


def _search_open_meteo(city, state):
    resp = requests.get(
        OPEN_METEO_URL,
        params={"name": city, "count": 20, "language": "en", "format": "json"},
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    places = []
    for r in resp.json().get("results") or []:
        if state and ((r.get("country_code") or "").upper() != "US"
                      or (r.get("admin1") or "").lower() != state.lower()):
            continue
        places.append({
            "label": _label(r["name"], r.get("admin1"), r.get("country_code")),
            "lat": float(r["latitude"]), "lon": float(r["longitude"]),
            "timezone": r.get("timezone"), "population": int(r.get("population") or 0),
        })
    return places


def _search_nominatim(city, state):
    query = f"{city}, {state}" if state else city
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": query, "format": "jsonv2", "addressdetails": 1, "limit": 8, "countrycodes": "us"},
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    places = []
    for r in resp.json():
        address = r.get("address", {})
        name = r.get("name") or address.get("city") or address.get("town") or address.get("village") or city
        places.append({
            "label": _label(name, address.get("state"), address.get("country_code")),
            "lat": float(r["lat"]), "lon": float(r["lon"]),
            "timezone": None, "population": 0,
        })
    return places


def search_places(text):
    """Matches for "City, ST" text, largest first: [{label, lat, lon, timezone, population}].
    Empty when nothing matches or both services are unavailable. Never raises."""
    city, state = parse_query(text)
    if not city:
        return []
    for search in (_search_open_meteo, _search_nominatim):
        try:
            places = search(city, state)
        except Exception:
            logger.exception("Place lookup failed (%s)", search.__name__)
            continue
        if places:
            return sorted(places, key=lambda p: -p["population"])
    return []

"""City/state/province -> coordinates (and time zone) lookup for the location box.

Primary source is Open-Meteo's free, keyless geocoding API (the app already
uses Open-Meteo for forecasts); it returns the state, coordinates, population
and IANA time zone. If it is unreachable or finds nothing, OpenStreetMap's
Nominatim search (already used for sky-condition defaults, with the same
descriptive User-Agent) is tried; it has no time zone, so that field is None.
Everything fails soft: errors become an empty result list.
"""
import logging
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

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


CA_PROVINCES = {
    "AB": "Alberta", "BC": "British Columbia", "MB": "Manitoba", "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador", "NS": "Nova Scotia", "NT": "Northwest Territories",
    "NU": "Nunavut", "ON": "Ontario", "PE": "Prince Edward Island", "QC": "Quebec",
    "SK": "Saskatchewan", "YT": "Yukon",
}
# Mexican states are written out in full: their short codes clash with Canada's (BC, NL...).
MX_STATES = [
    "Aguascalientes", "Baja California", "Baja California Sur", "Campeche", "Chiapas",
    "Chihuahua", "Coahuila", "Colima", "Mexico City", "Durango", "Guanajuato", "Guerrero",
    "Hidalgo", "Jalisco", "México", "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca",
    "Puebla", "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora", "Tabasco",
    "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas",
]
MX_ALIASES = {
    "CDMX": "Mexico City", "Distrito Federal": "Mexico City", "Ciudad de Mexico": "Mexico City",
    "Estado de Mexico": "México", "Edomex": "México", "Coahuila de Zaragoza": "Coahuila",
    "Michoacan de Ocampo": "Michoacán", "Veracruz de Ignacio de la Llave": "Veracruz",
}
COUNTRY_WORDS = {"usa": "US", "us": "US", "united states": "US", "canada": "CA", "mexico": "MX", "méxico": "MX"}
NORTH_AMERICA = {"US", "CA", "MX"}  # the app covers these three countries


def _norm(text):
    """Lowercase, accent-free, single-spaced: "Nuevo León" == "nuevo leon"."""
    text = unicodedata.normalize("NFKD", (text or "").strip().rstrip(".")).encode("ascii", "ignore").decode()
    return " ".join(text.lower().split())


# normalized name or abbreviation -> (region name as Open-Meteo spells it, country code)
_REGIONS = {}
for _abbr, _name in US_STATES.items():
    _REGIONS[_norm(_abbr)] = (_name, "US")
    _REGIONS[_norm(_name)] = (_name, "US")
for _abbr, _name in CA_PROVINCES.items():
    _REGIONS[_norm(_abbr)] = (_name, "CA")
    _REGIONS[_norm(_name)] = (_name, "CA")
for _name in MX_STATES:
    _REGIONS[_norm(_name)] = (_name, "MX")
for _alias, _name in MX_ALIASES.items():
    _REGIONS[_norm(_alias)] = (_name, "MX")
# Bare "Mexico" is the country, not the state of México, unless written "Estado de Mexico".
_REGIONS.pop("mexico")


def _find_region(token):
    """(region name, country code) for "oh", "Ohio", "ON", "Ontario", "Nuevo León"; else None."""
    return _REGIONS.get(_norm(token))


def _split_region(text):
    """(city, region_info_or_None) from "Columbus, OH", "Columbus OH" or "Columbus"."""
    if "," in text:
        city, _, tail = text.rpartition(",")
        region = _find_region(tail)
        if region:
            return city.strip(), region
        return " ".join(text.replace(",", " ").split()), None
    words = text.split()
    for n in (3, 2, 1):  # longest region name first ("North Carolina" before "Carolina")
        if len(words) > n and (region := _find_region(" ".join(words[-n:]))):
            return " ".join(words[:-n]), region
    return text, None


def parse_query(text):
    """(city, region_name_or_None, country_code_or_None) for a US, Canadian or Mexican place.

    Understands "Columbus, OH", "Toronto, Ontario", "Vancouver BC", "Monterrey, Nuevo León",
    plus a trailing country ("Toronto, Ontario, Canada", "Toronto Canada"). The country is
    None unless the text names one or the region implies it."""
    text = (text or "").strip(" ,")
    if not text:
        return "", None, None
    city, region = _split_region(text)
    country = None
    if not region:
        match = re.search(r"[,\s]+(usa|us|united states|canada|mexico|méxico)\s*$", text, flags=re.I)
        if match:
            country = COUNTRY_WORDS[match.group(1).lower()]
            text = text[:match.start()].strip(" ,")
            city, region = _split_region(text) if text else ("", None)
    if region:
        return city, region[0], country or region[1]
    return city, None, country


def _admin1_matches(admin1, region):
    return _norm(admin1) == _norm(region)


def matching_zone(zone, options, year=None):
    """The zone in `options` that keeps the same clock time as `zone` all year, else None.

    Geocoders return precise zone names (America/Detroit, America/Boise, America/Juneau)
    that the app's short dropdown doesn't list; this maps them to the listed zone with the
    same UTC offset in both winter and summer, so daylight saving time lines up too
    (Boise -> Denver, not Phoenix)."""
    if not zone:
        return None
    if zone in options:
        return zone
    try:
        tz = ZoneInfo(zone)
    except Exception:
        return None
    year = year or datetime.now().year
    dates = [datetime(year, 1, 15, 12), datetime(year, 7, 15, 12)]

    def offsets(z):
        return [d.replace(tzinfo=z).utcoffset() for d in dates]

    target = offsets(tz)
    for option in options:
        if offsets(ZoneInfo(option)) == target:
            return option
    return None


def _label(name, state, country_code):
    parts = [name]
    if state and state != name:
        parts.append(state)
    if country_code and country_code.upper() != "US":
        parts.append(country_code.upper())
    return ", ".join(parts)


def _search_open_meteo(city, region, country):
    resp = requests.get(
        OPEN_METEO_URL,
        params={"name": city, "count": 20, "language": "en", "format": "json"},
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    places = []
    for r in resp.json().get("results") or []:
        code = (r.get("country_code") or "").upper()
        if code not in (({country} if country else NORTH_AMERICA)):
            continue
        if region and not _admin1_matches(r.get("admin1"), region):
            continue
        places.append({
            "label": _label(r["name"], r.get("admin1"), code),
            "lat": float(r["latitude"]), "lon": float(r["longitude"]),
            "timezone": r.get("timezone"), "population": int(r.get("population") or 0),
        })
    return places


def _search_nominatim(city, region, country):
    query = f"{city}, {region}" if region else city
    codes = country.lower() if country else ",".join(sorted(c.lower() for c in NORTH_AMERICA))
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": query, "format": "jsonv2", "addressdetails": 1, "limit": 8, "countrycodes": codes},
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
    city, region, country = parse_query(text)
    if not city:
        return []
    for search in (_search_open_meteo, _search_nominatim):
        try:
            places = search(city, region, country)
        except Exception:
            logger.exception("Place lookup failed (%s)", search.__name__)
            continue
        if places:
            # An exact name match ("Vancouver") outranks a bigger partial one ("Vancouver Island").
            return sorted(places, key=lambda p: (_norm(p["label"].split(",")[0]) != _norm(city), -p["population"]))
    return []

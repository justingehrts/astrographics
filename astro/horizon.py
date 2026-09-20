"""Real terrain horizon silhouette, replacing the original fixed
procedural sine-wave "suburban trees" shape (identical regardless of the
observer's actual location) with one derived from real elevation data via
Open-Meteo's free, keyless Elevation API -- the same data provider the
sibling model-viewer app already uses for weather.

For each azimuth in view, samples points at increasing distance along
that bearing and takes the steepest elevation angle among them (a nearby
high point can block a farther low one, but not vice versa, so the
horizon at any azimuth is the max angle along its ray). Distances include
an Earth-curvature + typical-refraction correction.

Network calls can fail or be slow; callers should catch exceptions here
and fall back to a simpler silhouette rather than blocking the UI.
"""
import numpy as np
import requests

EARTH_RADIUS_KM = 6371.0
SAMPLE_DISTANCES_KM = (1, 3, 7, 15, 30, 60)
ELEVATION_API_URL = "https://api.open-meteo.com/v1/elevation"
REQUEST_TIMEOUT_S = 8
_BATCH_SIZE = 500


def _destination(lat_deg, lon_deg, bearing_deg, distance_km):
    """Great-circle destination point given a start, bearing, and distance."""
    lat1, lon1 = np.radians(lat_deg), np.radians(lon_deg)
    brng = np.radians(bearing_deg)
    d_r = distance_km / EARTH_RADIUS_KM
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d_r) + np.cos(lat1) * np.sin(d_r) * np.cos(brng))
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(d_r) * np.cos(lat1),
        np.cos(d_r) - np.sin(lat1) * np.sin(lat2),
    )
    return np.degrees(lat2), np.degrees(lon2)


def _query_elevations(lats, lons):
    out = []
    for start in range(0, len(lats), _BATCH_SIZE):
        chunk_lats = lats[start:start + _BATCH_SIZE]
        chunk_lons = lons[start:start + _BATCH_SIZE]
        resp = requests.get(
            ELEVATION_API_URL,
            params={
                "latitude": ",".join(f"{v:.5f}" for v in chunk_lats),
                "longitude": ",".join(f"{v:.5f}" for v in chunk_lons),
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        out.extend(resp.json()["elevation"])
    return out


def fetch_horizon_profile(lat, lon, azimuths_deg):
    """Returns an array of horizon altitude angles (degrees, >= 0), one
    per azimuth in `azimuths_deg`, derived from real terrain elevation
    around (lat, lon). Raises on network/parse failure."""
    azimuths_deg = np.asarray(azimuths_deg, dtype=float)

    ray_points = [
        (i, d, *_destination(lat, lon, az, d))
        for i, az in enumerate(azimuths_deg)
        for d in SAMPLE_DISTANCES_KM
    ]

    query_lats = [lat] + [p[2] for p in ray_points]
    query_lons = [lon] + [p[3] for p in ray_points]
    elevations = _query_elevations(query_lats, query_lons)

    observer_elev_m = elevations[0]
    ray_elevations = elevations[1:]

    horizon_deg = np.zeros(len(azimuths_deg))
    for (i, d, _, _), elev_m in zip(ray_points, ray_elevations):
        distance_m = d * 1000.0
        # Standard "dip of horizon" curvature-drop approximation with a
        # ~13% correction for typical atmospheric refraction.
        drop_m = 0.87 * distance_m ** 2 / (2.0 * EARTH_RADIUS_KM * 1000.0)
        angle_deg = np.degrees(np.arctan2(elev_m - observer_elev_m - drop_m, distance_m))
        horizon_deg[i] = max(horizon_deg[i], angle_deg)

    return np.clip(horizon_deg, 0.0, None)

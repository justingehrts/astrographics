"""Decorative horizon foreground (buildings / trees / open field), layered
on top of the real terrain baseline from astro/horizon.py and styled by
how light-polluted the scene is (the "Star Visibility Limit" slider) --
matching the same urban/suburban/rural/pristine-dark tiers already used
for the sky_conditions captions in streamlit_app.py.

This is intentionally decorative, not a real building-by-building render
of the observer's actual skyline (there's no free data source for that);
it's seeded from the location and view direction so a given view always
looks the same rather than flickering between generates, but two
different places with the same light-pollution tier will look similar in
*style*, not identically.
"""
import numpy as np

WINDOW_LIT_COLOR = "#f4efe2"


def _rng(lat, lon, bearing):
    seed = int(abs(lat * 10007 + lon * 7919 + bearing * 104729)) % (2 ** 32)
    return np.random.RandomState(seed)


def _skyline(x_space, rng, n_buildings, min_height, max_height, tower_chance):
    """A step-function building silhouette: random contiguous widths
    across the view, each with a random flat-roofed height."""
    span = x_space[-1] - x_space[0]
    widths = rng.uniform(0.6, 1.4, n_buildings)
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    edges = edges / edges[-1] * span + x_space[0]
    heights = rng.uniform(min_height, max_height, n_buildings)
    is_tower = rng.uniform(0, 1, n_buildings) < tower_chance
    heights[is_tower] *= rng.uniform(1.4, 1.9, is_tower.sum())
    bucket = np.clip(np.searchsorted(edges, x_space, side="right") - 1, 0, n_buildings - 1)
    return heights[bucket], edges, heights


def _window_points(rng, edges, heights, lit_fraction):
    """Scattered lit-window points inside each building's rectangle."""
    points = []
    for i, h in enumerate(heights):
        x0, x1 = edges[i], edges[i + 1]
        width = x1 - x0
        if width < 0.4 or h < 1.0:
            continue
        cols = max(1, int(width / 0.5))
        rows = max(1, int(h / 0.8))
        for c in range(cols):
            for r in range(rows):
                if rng.uniform(0, 1) < lit_fraction:
                    x = x0 + (c + 0.5) * width / cols + rng.uniform(-0.05, 0.05)
                    y = (r + 0.5) * h / rows
                    points.append((x, y))
    return points


def _tree_line(x_space, rng, alt_max):
    """Smooth, non-repeating organic canopy via a small sum of randomly
    phased/scaled sine harmonics, plus a few taller trees poking up --
    avoids the obviously-periodic look of a fixed 2-3 term sine sum."""
    span = x_space[-1] - x_space[0]
    rel = x_space - x_space[0]
    profile = np.full_like(x_space, 0.22 * alt_max)
    for k in range(5):
        freq = rng.uniform(0.4, 2.5) * (k + 1)
        phase = rng.uniform(0, 2 * np.pi)
        amp = (rng.uniform(0.3, 1.1) / (k + 1)) * 0.09 * alt_max
        profile += amp * np.sin(2 * np.pi * freq * rel / span + phase)
    for pos in rng.uniform(x_space[0], x_space[-1], max(3, int(span / 12))):
        width = rng.uniform(1.0, 2.2)
        profile += rng.uniform(0.10, 0.22) * alt_max * np.exp(-((x_space - pos) / width) ** 2)
    return np.clip(profile, 0.08 * alt_max, 0.6 * alt_max)


def _open_field(x_space, rng, alt_max):
    """Mostly flat, gently rolling ground -- a remote/open dark-sky site."""
    span = x_space[-1] - x_space[0]
    rel = x_space - x_space[0]
    freq = rng.uniform(0.3, 0.7)
    phase = rng.uniform(0, 2 * np.pi)
    profile = 0.06 * alt_max + 0.025 * alt_max * np.sin(2 * np.pi * freq * rel / span + phase)
    return np.clip(profile, 0.02 * alt_max, 0.12 * alt_max)


def foreground_profile(x_space, star_brightness, lat, lon, bearing, alt_max):
    """Returns (heights, window_points): `heights` (same shape as
    `x_space`) to add on top of the real terrain baseline, and a list of
    (x, y) points -- in the same units, y relative to the *building's*
    own base -- for lit windows to scatter on urban/suburban tiers
    (empty for tree/open-field tiers).

    Heights scale with `alt_max` (the "Max Altitude Shown" setting)
    rather than being fixed degree values, so the foreground occupies a
    consistent, deliberate fraction of the frame regardless of how
    zoomed-in/out the current view is."""
    rng = _rng(lat, lon, bearing)
    span = x_space[-1] - x_space[0]

    if star_brightness <= 1.5:
        n = max(6, int(span / 3.0))
        heights, edges, building_heights = _skyline(
            x_space, rng, n, min_height=0.14 * alt_max, max_height=0.32 * alt_max, tower_chance=0.18
        )
        windows = _window_points(rng, edges, building_heights, lit_fraction=0.35)
        return heights, windows

    if star_brightness <= 2.5:
        n = max(4, int(span / 5.0))
        heights, edges, building_heights = _skyline(
            x_space, rng, n, min_height=0.06 * alt_max, max_height=0.17 * alt_max, tower_chance=0.06
        )
        windows = _window_points(rng, edges, building_heights, lit_fraction=0.15)
        return heights, windows

    if star_brightness <= 3.5:
        return _tree_line(x_space, rng, alt_max), []

    return _open_field(x_space, rng, alt_max), []

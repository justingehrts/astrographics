"""Nearby-foreground silhouette of the horizon: realistic urban, suburban or
rural scenes drawn in front of the distant terrain profile from
astro/horizon.py (or alone when that lookup fails). The caller picks the scene
from how light-polluted the current view is (see `scene_for_visibility`): a
city skyline under bright urban skies, houses and trees in the suburbs, and
farmland with tree stands and barns under dark rural skies.

The art is AI-generated (Gemini): several single-row black-on-white strips per
scene, each about 80 degrees of sky wide. tools/clean_horizon_strip.py turns
the raw strips into the alpha masks in data/horizon/<scene>/ -- crisp edges
(blur and JPEG damage removed), windows filled, tree crowns reattached to a
trunk, and both ends trimmed to a low fence/lawn stretch. Every strip is
stored at the same scale (OUT_H px == TREE_HEIGHT_DEG), so a house or tree has
the same angular size wherever it is placed.

Because the ends are low, strips are chained in a seeded random order with no
mirroring (a mirrored house would read as obviously reused); each view starts
at a random horizontal offset so different locations and directions don't all
show the same crop.

Height is a *fixed* angular size, not scaled by `alt_max`: a real treeline at
some fixed distance subtends a fixed angle regardless of how much total sky
the view happens to show above it -- at a large alt_max (e.g. showing all the
way to zenith) it should read as a thin band near the bottom, not grow to fill
a constant fraction of the frame.
"""
import glob
import os

import numpy as np
from PIL import Image

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "horizon")

SCENES = ("urban", "suburban", "rural")

SILHOUETTE_COLOR_RGB = (6, 12, 20)  # matches the app's #060c14 foreground color

# A distant treeline realistically subtends about this many degrees; every
# strip's pixel height corresponds to it.
TREE_HEIGHT_DEG = 10.0

# Star Visibility Limit (the sidebar slider) at or below which each scene is
# used: 1.0-1.5 city, 2.0-3.0 suburb, 3.5 and up countryside.
URBAN_MAX_BRIGHTNESS = 1.5
SUBURBAN_MAX_BRIGHTNESS = 3.0


def scene_for_visibility(star_brightness):
    """The horizon scene that matches a Star Visibility Limit."""
    if star_brightness <= URBAN_MAX_BRIGHTNESS:
        return "urban"
    if star_brightness <= SUBURBAN_MAX_BRIGHTNESS:
        return "suburban"
    return "rural"


def load_scene_strips(scene):
    """Loads a scene's cleaned strips as a list of uint8 alpha arrays
    (height x width). Callers should cache this (e.g. via st.cache_resource)
    -- it's static data, loaded from disk once."""
    paths = sorted(glob.glob(os.path.join(_DATA_DIR, scene, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No horizon strips found for scene {scene!r} in {_DATA_DIR}")
    return [np.array(Image.open(path).convert("L")) for path in paths]


def _chain_order(n_strips, n_needed, rng):
    """Strip indices in a random order that never repeats a strip twice in a
    row (so a single strip never sits next to a copy of itself)."""
    order = []
    while len(order) < n_needed:
        batch = list(rng.permutation(n_strips))
        if order and n_strips > 1 and batch[0] == order[-1]:
            batch[0], batch[-1] = batch[-1], batch[0]
        order.extend(batch)
    return order


def tiled_treeline(strips, az_min, az_max, seed):
    """Returns (rgba_array, extent) ready to hand straight to
    `ax.imshow(rgba_array, extent=extent, ...)`: the scene's strips chained
    in a seeded random order and cropped to cover [az_min, az_max] at a fixed
    angular height (TREE_HEIGHT_DEG), independent of alt_max -- see the module
    docstring for why. The array is uint8 RGBA, silhouette-colored."""
    height = strips[0].shape[0]
    px_per_deg = height / TREE_HEIGHT_DEG
    span = az_max - az_min
    needed_px = max(1, int(np.ceil(span * px_per_deg)))

    rng = np.random.RandomState(seed)
    widths = [s.shape[1] for s in strips]
    # Enough strips to cover the span even after dropping a random starting offset.
    n_needed = int(np.ceil(needed_px / min(widths))) + 2
    order = _chain_order(len(strips), n_needed, rng)
    chain = np.concatenate([strips[i] for i in order], axis=1)

    offset = int(rng.randint(0, max(1, chain.shape[1] - needed_px)))
    alpha = chain[:, offset:offset + needed_px]

    rgba = np.empty(alpha.shape + (4,), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = SILHOUETTE_COLOR_RGB
    rgba[..., 3] = alpha

    extent = [az_min, az_max, 0, TREE_HEIGHT_DEG]
    return rgba, extent


def silhouette_top_profile(rgba, extent, opaque_alpha=0.5):
    """(azimuths, heights_deg) of the silhouette's top edge per pixel
    column of a `tiled_treeline` result -- how high the foreground blocks
    the sky at each azimuth. Row 0 is the top of the image (imshow's
    default 'upper' origin, as the app draws it)."""
    az_min, az_max, _, top_deg = extent
    h, w = rgba.shape[:2]
    alpha = rgba[..., 3]
    if alpha.dtype == np.uint8:
        alpha = alpha / 255.0
    opaque = alpha > opaque_alpha
    first_opaque_row = np.where(opaque.any(axis=0), opaque.argmax(axis=0), h)
    heights = (h - first_opaque_row) / h * top_deg
    azimuths = az_min + (np.arange(w) + 0.5) / w * (az_max - az_min)
    return azimuths, heights

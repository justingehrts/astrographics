"""Real tree-line/houses-line silhouette images, used for the horizon when
the real terrain elevation lookup (astro/horizon.py) fails or isn't
location-specific enough to draw on its own. Two variants are available --
a plain tree-line (data/tree_line_silhouette.png) and a suburban
houses+trees mix (data/houses_treeline_silhouette.png) -- and the caller
picks between them based on how light-polluted the current view is (the
houses variant for urban/suburban skies, the plain tree-line for rural
ones), since the former realistically wouldn't be the horizon for a dark
rural sky.

The source images were AI-generated (Gemini/ImageFX). The tree-line image
had its visible "AI-generated" corner badge cropped out and the empty
transparent sky trimmed off, then was downsampled for reasonable tiling
memory use. The houses+trees image had no badge; it was cropped to its
sharper of two generated variants and had its white background keyed to
transparency by luminance.

Since the image has a fixed pixel width but the app's field of view is
user-adjustable (30-180deg), it's tiled horizontally to cover whatever
span is currently in view, mirroring alternate copies (so the shape is
continuous at each seam instead of jumping) and starting from a
per-view random horizontal offset (so different locations/directions
don't all show the exact same crop).

Height is a *fixed* angular size, not scaled by `alt_max`: a real
treeline at some fixed distance subtends a fixed angle regardless of
how much total sky the view happens to show above it -- at a large
alt_max (e.g. showing all the way to zenith) it should read as a thin
band near the bottom, not grow to fill a constant fraction of the
frame. (This is the opposite of the earlier skyline work, where fixed
degree sizes for buildings looked wrong at different zoom levels --
that was about a stylized foreground meant to fill the frame
deliberately, not a real object with a physical angular size.)
"""
import os

import matplotlib.image as mpimg
import numpy as np

_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tree_line_silhouette.png")
_HOUSES_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "houses_treeline_silhouette.png")

SILHOUETTE_COLOR_RGB = (6 / 255.0, 12 / 255.0, 20 / 255.0)  # matches the app's #060c14 foreground color

# A distant treeline realistically subtends about this many degrees.
# Fixed regardless of alt_max -- see the module docstring. Shared by both
# image variants for visual consistency when switching between them.
TREE_HEIGHT_DEG = 10.0


def load_treeline_image():
    """Loads the plain tree-line source RGBA image as a float array in
    [0, 1]. Callers should cache this (e.g. via st.cache_resource) -- it's
    static data, loaded from disk once."""
    return mpimg.imread(_IMAGE_PATH)


def load_houses_treeline_image():
    """Loads the suburban houses+trees source RGBA image as a float array
    in [0, 1]. Callers should cache this (e.g. via st.cache_resource) --
    it's static data, loaded from disk once."""
    return mpimg.imread(_HOUSES_IMAGE_PATH)


def _recolored(rgba):
    out = rgba.copy()
    out[..., 0] = SILHOUETTE_COLOR_RGB[0]
    out[..., 1] = SILHOUETTE_COLOR_RGB[1]
    out[..., 2] = SILHOUETTE_COLOR_RGB[2]
    return out


def tiled_treeline(source_rgba, az_min, az_max, seed):
    """Returns (rgba_array, extent) ready to hand straight to
    `ax.imshow(rgba_array, extent=extent, ...)`: a horizontally tiled,
    recolored strip of the source tree image sized to cover
    [az_min, az_max] at a fixed angular height (TREE_HEIGHT_DEG),
    independent of alt_max -- see the module docstring for why."""
    h, w = source_rgba.shape[:2]
    aspect = w / h

    tile_height_deg = TREE_HEIGHT_DEG
    tile_width_deg = tile_height_deg * aspect
    px_per_deg = w / tile_width_deg

    span = az_max - az_min
    needed_px = max(1, int(np.ceil(span * px_per_deg)))

    n_tiles = int(np.ceil((needed_px + w) / w)) + 1
    recolored = _recolored(source_rgba)
    mirrored = recolored[:, ::-1, :]
    strip = np.concatenate([recolored if i % 2 == 0 else mirrored for i in range(n_tiles)], axis=1)

    rng = np.random.RandomState(seed)
    max_offset = max(1, strip.shape[1] - needed_px)
    offset = int(rng.randint(0, max_offset))
    cropped = strip[:, offset:offset + needed_px, :]

    extent = [az_min, az_max, 0, tile_height_deg]
    return cropped, extent

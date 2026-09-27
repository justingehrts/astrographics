"""Real tree-line silhouette image, used for the horizon when the real
terrain elevation lookup (astro/horizon.py) fails or isn't location-
specific enough to draw on its own.

The source image (data/tree_line_silhouette.png) was AI-generated
(Gemini/ImageFX), with its visible "AI-generated" corner badge cropped
out and the empty transparent sky trimmed off, then downsampled for
reasonable tiling memory use.

Since the image has a fixed pixel width but the app's field of view is
user-adjustable (30-180deg), it's tiled horizontally to cover whatever
span is currently in view, mirroring alternate copies (so the shape is
continuous at each seam instead of jumping) and starting from a
per-view random horizontal offset (so different locations/directions
don't all show the exact same crop). Height scales with `alt_max`
(matching the lesson from the earlier skyline work: fixed absolute
degree sizes look wrong at different zoom levels) via the image's own
aspect ratio, so trees are never squashed or stretched.
"""
import os

import matplotlib.image as mpimg
import numpy as np

_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tree_line_silhouette.png")

SILHOUETTE_COLOR_RGB = (6 / 255.0, 12 / 255.0, 20 / 255.0)  # matches the app's #060c14 foreground color

# How wide one un-tiled copy of the image reads as, in degrees, scaled by
# alt_max -- the image's own aspect ratio then determines its height. The
# cropped source image is ~6.3:1 (wide, low content band), so this needs
# to be noticeably larger than it would for a squarer image to reach a
# comparable on-screen height (~15% of alt_max).
TILE_WIDTH_FRACTION_OF_ALT_MAX = 0.95


def load_treeline_image():
    """Loads the source RGBA image as a float array in [0, 1]. Callers
    should cache this (e.g. via st.cache_resource) -- it's static data,
    loaded from disk once."""
    return mpimg.imread(_IMAGE_PATH)


def _recolored(rgba):
    out = rgba.copy()
    out[..., 0] = SILHOUETTE_COLOR_RGB[0]
    out[..., 1] = SILHOUETTE_COLOR_RGB[1]
    out[..., 2] = SILHOUETTE_COLOR_RGB[2]
    return out


def tiled_treeline(source_rgba, az_min, az_max, alt_max, seed):
    """Returns (rgba_array, extent) ready to hand straight to
    `ax.imshow(rgba_array, extent=extent, ...)`: a horizontally tiled,
    recolored strip of the source tree image sized to cover
    [az_min, az_max] at a height proportional to alt_max."""
    h, w = source_rgba.shape[:2]
    aspect = w / h

    tile_width_deg = TILE_WIDTH_FRACTION_OF_ALT_MAX * alt_max
    tile_height_deg = tile_width_deg / aspect
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

"""Well-known asterisms: patterns the audience recognizes that either cross
constellation lines (Summer Triangle, Winter Hexagon) or get lost inside a
full constellation figure (the Big Dipper is just the tail of Ursa Major's
bear). A small hand-made list of Hipparcos star numbers -- no external data.

Each asterism has `paths` (solid lines) and optional `guides` (dashed lines,
for "follow this line to find another star" pointers). A path is an ordered
tuple of Hipparcos IDs joined consecutively; repeating the first ID closes the
shape.
"""

ASTERISMS = [
    {
        "name": "Big Dipper",
        "paths": [
            # Bowl (Dubhe, Merak, Phecda, Megrez), then the handle (Alioth, Mizar, Alkaid).
            (54061, 53910, 58001, 59774, 54061),
            (59774, 62956, 65378, 67301),
        ],
        # The "pointer" stars Merak and Dubhe, extended to Polaris. The guide starts at
        # Dubhe because Merak-Dubhe is already a solid bowl edge; dashing it too would
        # draw dashes over the solid line.
        "guides": [(54061, 11767)],
    },
    {
        "name": "Summer Triangle",
        "paths": [(91262, 102098, 97649, 91262)],  # Vega, Deneb, Altair
        "guides": [],
    },
    {
        "name": "Winter Triangle",
        "paths": [(27989, 32349, 37279, 27989)],  # Betelgeuse, Sirius, Procyon
        "guides": [],
    },
    {
        "name": "Winter Hexagon",
        # Rigel, Sirius, Procyon, Pollux, Capella, Aldebaran.
        "paths": [(24436, 32349, 37279, 37826, 24608, 21421, 24436)],
        "guides": [],
    },
    {
        "name": "Orion's Belt",
        "paths": [(26727, 26311, 25930)],  # Alnitak, Alnilam, Mintaka
        "guides": [],
    },
    {
        "name": "Great Square",
        "paths": [(113963, 113881, 677, 1067, 113963)],  # Markab, Scheat, Alpheratz, Algenib
        "guides": [],
    },
]


def lines(asterism):
    """[(hip_a, hip_b, dashed), ...] for every line of one asterism."""
    out = []
    for key, dashed in (("paths", False), ("guides", True)):
        for path in asterism[key]:
            out.extend((a, b, dashed) for a, b in zip(path, path[1:]))
    return out


def all_hips():
    """Every Hipparcos ID any asterism uses."""
    return {hip for asterism in ASTERISMS for key in ("paths", "guides")
            for path in asterism[key] for hip in path}

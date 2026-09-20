"""Constellation stick figures, drawn as connected lines through the
Hipparcos catalog already loaded for the star field.

Line data: Marc van der Sluys, "ConstellationLines" (CC BY 4.0), with its
Bright Star Catalogue (HR) star numbers converted to Hipparcos (HIP)
numbers via the HYG Database's hip/hr cross-reference (CC BY-SA 4.0) --
bundled at data/constellation_lines.csv; see
data/CONSTELLATION_LINES_ATTRIBUTION.md for the full citation.
"""
import csv
import os

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "constellation_lines.csv")


def load_constellation_segments(path=_DATA_PATH):
    """Returns a list of (hip_a, hip_b) Hipparcos-ID pairs, one per line
    segment to draw, derived from each constellation's ordered
    "connect the dots" star path in the bundled dataset."""
    segments = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header row
        for row in reader:
            hip_ids = [int(v) for v in row[2:] if v.strip()]
            segments.extend(zip(hip_ids, hip_ids[1:]))
    return segments

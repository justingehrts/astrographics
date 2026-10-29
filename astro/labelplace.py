"""Nudge text labels clear of the lines and other labels already on an axes.

Used for the constellation and asterism names, which can land on a figure's
lines or on each other (e.g. "Big Dipper" over "Ursa Major"). Everything is
measured in display pixels at the figure's working dpi, so the result holds at
any export resolution (the whole figure scales together)."""
import numpy as np

_PAD_PX = 3.0          # clear space kept around a label
_LINE_STEP_PX = 3.0    # spacing of the sample points along each line
_ALT_OFFSETS_DEG = (0.0, 1.5, 3.0, -1.5, -3.0, 4.5, 6.0)
_AZ_OFFSETS_WIDTHS = (0.0, -0.6, 0.6, -1.1, 1.1, -1.7, 1.7)  # in label widths


def _line_samples(ax):
    """Pixel-space points along every drawn line, for overlap tests."""
    pts = []
    for line in ax.lines:
        xy = np.asarray(line.get_xydata(), dtype=float)
        if len(xy) < 2 or not np.isfinite(xy).all():
            continue
        disp = ax.transData.transform(xy)
        for a, b in zip(disp[:-1], disp[1:]):
            n = max(int(np.hypot(*(b - a)) / _LINE_STEP_PX), 1)
            t = np.linspace(0.0, 1.0, n + 1)[:, None]
            pts.append(a + (b - a) * t)
    return np.vstack(pts) if pts else np.empty((0, 2))


def place_clear(ax, renderer, text, above_horizon):
    """Move `text` (an existing ax.text artist) to the nearby spot that overlaps
    the least: no other label, no drawn line, inside the frame, above the
    skyline. `above_horizon(az_array, alt_array)` -> bool. Returns the artist."""
    others = [t.get_window_extent(renderer) for t in ax.texts if t is not text and t.get_visible()]
    samples = _line_samples(ax)
    frame = ax.bbox
    x0, y0 = text.get_position()
    box = text.get_window_extent(renderer)
    width_data = abs(ax.transData.inverted().transform([(box.x1, box.y0)])[0][0]
                     - ax.transData.inverted().transform([(box.x0, box.y0)])[0][0])

    best, best_score = (x0, y0), None
    for d_alt in _ALT_OFFSETS_DEG:
        for d_az in _AZ_OFFSETS_WIDTHS:
            text.set_position((x0 + d_az * width_data, y0 + d_alt))
            b = text.get_window_extent(renderer)
            padded = (b.x0 - _PAD_PX, b.y0 - _PAD_PX, b.x1 + _PAD_PX, b.y1 + _PAD_PX)
            score = 0.0
            if b.x0 < frame.x0 or b.x1 > frame.x1 or b.y1 > frame.y1:
                score += 1e6
            corners = ax.transData.inverted().transform([(b.x0, b.y0), ((b.x0 + b.x1) / 2, b.y0), (b.x1, b.y0)])
            if not np.all(above_horizon(corners[:, 0], corners[:, 1])):
                score += 1e5
            score += 1e3 * sum(
                1 for o in others
                if padded[0] < o.x1 and padded[2] > o.x0 and padded[1] < o.y1 and padded[3] > o.y0
            )
            if len(samples):
                inside = ((samples[:, 0] > padded[0]) & (samples[:, 0] < padded[2])
                          & (samples[:, 1] > padded[1]) & (samples[:, 1] < padded[3]))
                score += float(inside.sum())
            if best_score is None or score < best_score:
                best, best_score = (x0 + d_az * width_data, y0 + d_alt), score
            if score == 0.0:
                text.set_position(best)
                return text
    text.set_position(best)
    return text

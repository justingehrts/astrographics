"""Turns raw Gemini horizon strips into the cleaned alpha masks the app ships
in data/horizon/<scene>/.

Raw strips are black silhouettes on white, one row, ~10 deg of sky tall. They
arrive blurred, JPEG-damaged, or with white keylines/windows, and sometimes with
tree crowns that have no trunk. This tool, per strip:

  1. upsamples 4x (bicubic) and thresholds at 50% gray -- blur is symmetric, so
     edges stay where they were;
  2. closes hairline white cracks between overlapping objects;
  3. fills window/door/garage holes (tiny, or near-rectangular), keeping the
     irregular sky gaps between trunks and canopies;
  4. lightly opens the shape (smooths ragged edges), drops specks and anything
     not connected to the ground band, and gives crowns that float in the sky a
     trunk down to whatever is below them;
  5. rural scenes keep their fences: the bottom band is cleaned with gentler
     settings (stronger closing/opening would erase the rails);
  6. trims each end back to a low stretch (fence/hedge/lawn) so strips can be
     chained in any order without a tree cut in half at the join;
  7. resizes to a common scale (OUT_H px = 10 deg) and writes a single-channel
     PNG that is the silhouette's alpha.

usage: clean_horizon_strip.py SCENE OUT_DIR SRC [SRC ...]
(SCENE is urban, suburban or rural; sources are untrusted input, decoded as
pixels only.)
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SS = 4              # supersampling factor
OUT_H = 480         # output height in px == 10 deg (48 px/deg), same for every scene
THRESHOLD = 128
TRUNK_PX = 2.4      # trunk width for re-attached crowns (1x px)
TRUNK_MAX_PX = 40   # never extend a trunk further than this (1x px)
FENCE_BAND = 0.22   # rural: bottom fraction of the strip cleaned gently (fences)
END_LOWS = (0.30, 0.45, 0.60)  # "clear" = below this fraction of H; lowest that works wins
END_WINDOW = 0.20   # look for a clear stretch within this fraction of each end
END_RUN = 0.015     # ... at least this wide (fraction of strip width)

# boxy_max: near-rectangular holes up to this fraction of H^2 are filled
# (windows, doors); hole_frac: any hole smaller than this is filled.
STRONG = dict(crack=2.2, hole_frac=0.0055, boxy_max=0.045, speck=40, open=1.0)
# Rural art has fine canopy detail and thin fence rails that STRONG smears,
# but still needs its barn trim lines and windows closed, so it gets a
# milder version above the fence band and GENTLE inside it.
RURAL = dict(crack=1.8, hole_frac=0.0008, boxy_max=0.012, speck=12, open=0.4)
GENTLE = dict(crack=1.2, hole_frac=0.0008, boxy_max=0.0, speck=12, open=0.4)
PRESETS = {"urban": STRONG, "suburban": STRONG, "rural": RURAL}


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def attach_floating(m):
    """From the lowest pixel of each piece not touching the ground, extend a
    trunk straight down until it meets something. Re-labels after every
    attachment so a crown stacked above another chains down to the ground."""
    h, w = m.shape
    st8 = np.ones((3, 3), int)
    half = max(1, int(round(TRUNK_PX * SS / 2)))
    attached = 0
    for _ in range(8):
        lab, _n = ndi.label(m, structure=st8)
        ground = set(np.unique(lab[-SS:, :])) - {0}
        pieces = [(i, sl) for i, sl in enumerate(ndi.find_objects(lab), 1) if i not in ground]
        if not pieces:
            break
        pieces.sort(key=lambda t: -t[1][0].stop)
        progressed = False
        for i, (ys, xs) in pieces:
            piece = lab[ys, xs] == i
            cols = np.where(piece[piece.shape[0] - 1, :])[0]
            cx = xs.start + int(np.mean(cols))
            y = ys.stop
            limit = min(h, y + TRUNK_MAX_PX * SS)
            while y < limit and not (lab[y, max(0, cx - 1):cx + 2] != 0).any():
                y += 1
            if y < limit:
                m[ys.stop - 1:y + 1, max(0, cx - half):cx + half] = True
                attached += 1
                progressed = True
                break
        if not progressed:
            break
    return m, attached


def clean_mask(gray, p):
    h, w = gray.shape
    big = np.array(Image.fromarray(gray).resize((w * SS, h * SS), Image.BICUBIC))
    m = big < THRESHOLD
    # Morphology treats the border as empty and would eat the outermost
    # pixels; replicate the edge outward first and crop back at the end.
    pad = 8 * SS
    m = np.pad(m, pad, mode="edge")

    m = ndi.binary_closing(m, structure=disk(max(1, int(round(p["crack"] * SS / 2)))))

    holes = ndi.binary_fill_holes(m) & ~m
    hl, hn = ndi.label(holes)
    if hn:
        hs = h * SS
        fill = np.zeros(hn + 1, bool)
        for i, sl in enumerate(ndi.find_objects(hl), 1):
            area = int((hl[sl] == i).sum())
            bbox = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
            small = area < p["hole_frac"] * hs ** 2
            boxy = area / bbox >= 0.8 and area < p["boxy_max"] * hs ** 2
            fill[i] = small or boxy
        m = m | fill[hl]

    m = ndi.binary_opening(m, structure=disk(max(1, int(round(p["open"] * SS / 2)))))

    near = ndi.binary_dilation(m, structure=disk(3 * SS))
    lab, _n = ndi.label(near)
    m = m & np.isin(lab, list(set(np.unique(lab[-SS:, :])) - {0}))

    lab2, n2 = ndi.label(m)
    sizes = ndi.sum(m, lab2, index=np.arange(1, n2 + 1))
    keep = np.zeros(n2 + 1, bool)
    keep[1:] = sizes >= p["speck"] * SS * SS
    m, attached = attach_floating(keep[lab2])
    return m[pad:-pad, pad:-pad], attached


def clean_strip(src, scene):
    gray = np.array(Image.open(src).convert("L"))
    m, attached = clean_mask(gray, PRESETS[scene])
    if scene == "rural":
        fine, _ = clean_mask(gray, GENTLE)
        band = int(FENCE_BAND * m.shape[0])
        m[-band:] = fine[-band:]
    return m, attached


def trim_ends(m):
    """Cut each end back to the clear (low) stretch nearest to it, trying
    progressively looser definitions of "low". Returns (trimmed mask,
    (left_level, right_level)); a level is None if no stretch was found."""
    h, w = m.shape
    top = np.where(m.any(0), m.argmax(0), h)
    height = (h - top) / h
    run = max(1, int(END_RUN * w))
    win = int(END_WINDOW * w)

    def find(cols):
        """(middle of the first run of >= `run` clear columns, level used) in
        `cols`, an index sequence ordered from the strip's edge inward."""
        for level in END_LOWS:
            count = 0
            for k, c in enumerate(cols):
                count = count + 1 if height[c] < level else 0
                if count >= run:
                    return cols[k - run // 2], level
        return None, None

    left, llev = find(list(range(0, win)))
    right, rlev = find(list(range(w - 1, w - 1 - win, -1)))
    lo = left if left is not None else 0
    hi = right + 1 if right is not None else w
    return m[:, lo:hi], (llev, rlev)


def to_alpha(m):
    h, w = m.shape
    soft = ndi.gaussian_filter(m.astype(float), 0.9)
    img = Image.fromarray((soft * 255).astype("uint8"))
    new_w = max(1, int(round(w * OUT_H / h)))
    return np.array(img.resize((new_w, OUT_H), Image.LANCZOS))


def main(scene, out_dir, sources):
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for src in sources:
        m, attached = clean_strip(src, scene)
        m, (llev, rlev) = trim_ends(m)
        name = os.path.basename(src)
        if llev is None or rlev is None:
            # A tall tree at an end would be cut in half (or leave a cliff)
            # wherever the strip is chained to another.
            print(f"{name}: SKIPPED, no clear stretch at the {'left' if llev is None else 'right'} end")
            continue
        written += 1
        a = to_alpha(m)
        out = os.path.join(out_dir, f"{scene}_{written:02d}.png")
        Image.fromarray(a, "L").save(out, optimize=True)
        print(f"{name} -> {out} {a.shape[1]}x{a.shape[0]}  attached={attached}  "
              f"clear-end level: left={llev} right={rlev}")


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] not in PRESETS:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])

"""
Image scale (micrometers per pixel) for pics2graph.

The scale usually comes from another picture taken with the same camera and
objective, with the microscope's overlay (a scale bar and its length, e.g.
"100 um"), since the tiles of a mosaic are saved without it.

* ``find_scale_bar(img)`` finds the bar of the overlay: the longest straight,
  thin, horizontal (or vertical) run of a single saturated color (yellow,
  green, red...) or of pure white.
* ``pixel_size_from_scale_bar(img_or_path, bar_um)`` gives um / px from it.
* ``read_pixel_size(path)`` reads a calibration stored in a TIFF (ImageJ /
  Fiji "Set Scale", or the TIFF resolution tags), or returns None.

A mosaic stitched from tiles reduced by a factor ``d`` (``--stitch-downsample``)
has pixels ``d`` times larger than those of the scale-bar picture.
"""

import cv2
import numpy as np

_MICRONS = {"um": 1.0, "µm": 1.0, "μm": 1.0, "micron": 1.0, "microns": 1.0,
            "micrometer": 1.0, "micrometre": 1.0, "nm": 1e-3, "mm": 1e3,
            "cm": 1e4, "m": 1e6}


def _read(img_or_path):
    if isinstance(img_or_path, str):
        data = np.fromfile(img_or_path, dtype=np.uint8)   # works with non-ASCII paths
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read the image: {img_or_path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        return img
    return np.asarray(img_or_path)


def _runs(mask, min_length):
    """All horizontal runs of True at least ``min_length`` long: (y, x0, x1)."""
    out = []
    for y in np.flatnonzero(mask.sum(axis=1) >= min_length):
        d = np.diff(np.r_[0, mask[y].astype(np.int8), 0])
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        for x0, x1 in zip(s, e):
            if x1 - x0 >= min_length:
                out.append((int(y), int(x0), int(x1) - 1))
    return out


def _best_bar(mask, rgb, min_length, max_thickness):
    """
    Longest run of ``mask`` that looks like an overlay bar: thin, of uniform
    color, not touching the image border. Returns (length, y, x0, x1) or None.
    """
    H, W = mask.shape
    best = None
    for y, x0, x1 in sorted(_runs(mask, min_length), key=lambda r: r[1] - r[2]):
        L = x1 - x0 + 1
        if best is not None and L <= best[0]:
            break
        if x0 == 0 or x1 == W - 1 or y == 0 or y == H - 1:
            continue
        y0, y1 = max(y - max_thickness - 2, 0), min(y + max_thickness + 3, H)
        if (mask[y0:y1, x0:x1 + 1].mean(axis=1) > 0.8).sum() > max_thickness:
            continue                       # not thin: a filled area
        if rgb is not None and rgb[y, x0:x1 + 1].astype(float).std(axis=0).max() > 20:
            continue                       # not one flat color: image content
        best = (L, y, x0, x1)
    return best


def find_scale_bar(img, min_length=20, max_thickness=12):
    """
    The scale bar of a microscope overlay, as ``(x0, y0, x1, y1, length_px)``
    (ends of the bar, in px; ``length_px`` between their centers), or None.

    Looks for the longest thin straight run, horizontal or vertical, of one
    flat, strongly saturated color (any hue) and, if there is none, of pure
    white.
    """
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        rgb = None
        candidates = [img >= 250]
    else:
        rgb = img[..., :3]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
        colorful = (s >= 200) & (v >= 200)
        candidates = []
        if colorful.any():
            # one mask per dominant hue, so that a bar is a single color
            hist = np.bincount(h[colorful] // 6, minlength=31)
            for b in np.argsort(hist)[::-1][:4]:
                if hist[b] < min_length:
                    break
                hue = (h // 6 >= b - 1) & (h // 6 <= b + 1)
                candidates.append(colorful & hue)
        candidates.append(rgb.min(axis=2) >= 250)
    for m in candidates:
        found = []
        r = _best_bar(m, rgb, min_length, max_thickness)
        if r:
            L, y, x0, x1 = r
            found.append((L, (x0, y, x1, y)))
        r = _best_bar(m.T, None if rgb is None else rgb.transpose(1, 0, 2),
                      min_length, max_thickness)
        if r:
            L, x, y0, y1 = r
            found.append((L, (x, y0, x, y1)))
        if found:
            L, (x0, y0, x1, y1) = max(found)
            return (x0, y0, x1, y1, float(L - 1))   # between the end pixel centers
    return None


def pixel_size_from_scale_bar(img_or_path, bar_um, downsample=1):
    """
    Micrometers per pixel from the scale bar of an overlay picture whose bar
    is ``bar_um`` micrometers long. Returns ``(pixel_size, bar)`` with ``bar``
    as given by ``find_scale_bar``; raises ValueError if no bar is found.
    """
    bar = find_scale_bar(_read(img_or_path))
    if bar is None:
        raise ValueError("No scale bar found in the image.")
    return bar_um / bar[4] * downsample, bar


def read_pixel_size(path):
    """
    Micrometers per pixel stored in a TIFF (ImageJ calibration or resolution
    tags with a physical unit), or None if there is none.
    """
    if not str(path).lower().endswith((".tif", ".tiff")):
        return None
    try:
        import tifffile
    except ImportError:
        return None
    try:
        with tifffile.TiffFile(path) as t:
            page = t.pages[0]
            tags = page.tags
            xr = tags.get("XResolution")
            if xr is None:
                return None
            num, den = xr.value
            if not num or not den:
                return None
            px_per_unit = num / den
            ij = t.imagej_metadata or {}
            unit = ij.get("unit")
            if unit:
                unit = unit.encode("latin-1", "ignore").decode("unicode_escape", "ignore") \
                    if "\\u" in unit else unit
                factor = _MICRONS.get(unit.strip().lower())
                if factor is None:
                    return None
                return factor / px_per_unit
            ru = tags.get("ResolutionUnit")
            ru = int(ru.value) if ru is not None else 2
            if ru == 3:      # centimeter
                return 1e4 / px_per_unit
            # inch (2) is the default of most writers (72 / 300 dpi): not a
            # physical calibration of a micrograph
            return None
    except Exception:
        return None

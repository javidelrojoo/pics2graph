"""
electrode_registration
======================

Locate the electrodes of a *later* photograph on an image of the network that
already exists (typically the stitched mosaic).

The usual case: the network is mapped first (e.g. a 10x mosaic), the
electrodes are deposited afterwards, and a new photo is taken, often at
another magnification (e.g. 5x), slightly rotated and shifted. The electrodes
are visible only in that photo, but the graph is extracted from the mosaic.

What it does:

1. **Registration.** Finds the same features of the wire network in both
   images (SIFT keypoints) and fits, with RANSAC, the similarity transform
   (scale + rotation + translation) that maps the photo onto the image. The
   scale can be given (e.g. 2 for a 5x photo on a 10x mosaic) or searched
   automatically.
2. **Detection.** Finds the two electrodes in the photo by a smoothed color
   or brightness feature ("warm": red minus blue, "bright", "dark", or "auto",
   which picks the one that best separates two classes) and keeps the two
   largest regions.
3. **Transfer.** Warps both masks into the coordinates of the image, so they
   can be used directly as ``<name>_electrode_mask1.tif`` / ``_mask2.tif``.

The detection is a starting point: check the masks (``check_image``, or the
GUI) and touch them up if needed. The registration itself is usually
sub-pixel accurate when the photo shows enough of the network.

Example
-------
    from electrode_registration import electrodes_from_photo
    vin, vout, info = electrodes_from_photo(mosaic, photo, scale=2)

Command line
------------
    python electrode_registration.py JTA_Ag_O3.tif JTA_Ag_O3_electrodes.JPG --scale 2
    python electrode_registration.py -h
"""

import argparse
import math
import os

import cv2
import numpy as np
from scipy import ndimage

FEATURES = ("auto", "warm", "bright", "dark")
AUTO_SCALES = (1.0, 2.0, 0.5, 3.0, 1 / 3, 4.0, 0.25)


# =============================================================================
#  I/O helpers (non-ASCII paths on Windows)
# =============================================================================
def imread(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


def imwrite(path, img, params=()):
    ext = os.path.splitext(path)[1] or ".tif"
    ok, buf = cv2.imencode(ext, img, list(params))
    if not ok:
        raise IOError(f"Could not write {path}")
    buf.tofile(path)


def _gray(img):
    """uint8 grayscale from a gray or BGR(A) image of any dtype."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., :3]
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _enhance(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(gray)


# =============================================================================
#  1. Registration
# =============================================================================
def _match(photo_g, image_g, scale, sift, ratio):
    """SIFT matches between the photo and the image, brought to similar resolution."""
    # shrink whichever image is finer, never upsample
    if scale >= 1:
        fa, fi = 1.0, 1.0 / scale
    else:
        fa, fi = scale, 1.0
    a = photo_g if fa == 1 else cv2.resize(photo_g, None, fx=fa, fy=fa, interpolation=cv2.INTER_AREA)
    b = image_g if fi == 1 else cv2.resize(image_g, None, fx=fi, fy=fi, interpolation=cv2.INTER_AREA)
    ka, da = sift.detectAndCompute(_enhance(a), None)
    kb, db = sift.detectAndCompute(_enhance(b), None)
    if da is None or db is None or len(ka) < 3 or len(kb) < 3:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), fi
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    pa = np.float32([ka[g.queryIdx].pt for g in good]).reshape(-1, 2) / fa
    pb = np.float32([kb[g.trainIdx].pt for g in good]).reshape(-1, 2) / fi
    return pa, pb, fi


def register(photo, image, scale=None, n_features=40000, ratio=0.8,
             ransac_px=6.0, min_inliers=25, verbose=False):
    """
    Similarity transform that maps ``photo`` onto ``image``.

    Parameters
    ----------
    photo, image : np.ndarray
        Gray or BGR images (as read by OpenCV).
    scale : float, optional
        Approximate image pixels per photo pixel (e.g. 2 for a 5x photo on a
        10x mosaic). If None, several scales are tried and the best kept.
    n_features : int
        Maximum SIFT keypoints per image.
    ratio : float
        Lowe's ratio test for the matches.
    ransac_px : float
        RANSAC tolerance, in pixels of the image.
    min_inliers : int
        Minimum consistent matches to accept the result.

    Returns
    -------
    dict with ``M`` (2x3 float matrix, photo -> image), ``scale``,
    ``rotation_deg``, ``inliers``, ``matches``, ``residual_median`` and
    ``residual_p90`` (image pixels) and ``tried`` (scale -> inliers).
    """
    pg, ig = _gray(photo), _gray(image)
    sift = cv2.SIFT_create(nfeatures=n_features)
    candidates = [float(scale)] if scale else list(AUTO_SCALES)
    best, tried = None, {}
    for s in candidates:
        pa, pb, fi = _match(pg, ig, s, sift, ratio)
        if len(pa) < 3:
            tried[round(s, 3)] = 0
            continue
        M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_px,
                                             maxIters=20000, confidence=0.999)
        n_in = int(inl.sum()) if inl is not None else 0
        tried[round(s, 3)] = n_in
        if verbose:
            print(f"  scale {s:.3g}: {len(pa)} matches, {n_in} consistent")
        if M is not None and (best is None or n_in > best[1]):
            best = (M, n_in, len(pa), pa[inl.ravel() == 1], pb[inl.ravel() == 1])
        if scale is None and n_in >= 20 * min_inliers:
            break                                   # clearly found, stop searching
    if best is None or best[1] < min_inliers:
        raise RuntimeError(
            f"Could not align the photo with the image (best: {0 if best is None else best[1]} "
            f"consistent matches, need {min_inliers}; tried scales {tried}). Check that both "
            "show the same region of the network, and give the approximate scale "
            "(image px per photo px) if the automatic search fails.")
    M, n_in, n_match, ia, ib = best
    res = np.linalg.norm(ia @ M[:, :2].T + M[:, 2] - ib, axis=1)
    return dict(M=M, scale=float(math.hypot(M[0, 0], M[1, 0])),
                rotation_deg=float(math.degrees(math.atan2(M[1, 0], M[0, 0]))),
                inliers=n_in, matches=n_match,
                residual_median=float(np.median(res)), residual_p90=float(np.percentile(res, 90)),
                tried=tried)


# =============================================================================
#  2. Electrode detection in the photo
# =============================================================================
def _feature_map(photo_bgr, feature, sigma):
    img = photo_bgr.astype(np.float32)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    f = {"warm": r - b, "bright": gray, "dark": -gray}[feature]
    return cv2.GaussianBlur(f, (0, 0), sigma), cv2.GaussianBlur(gray, (0, 0), sigma)


def _otsu(values):
    """Otsu threshold and separability (between-class / total variance)."""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        return lo, 0.0
    v8 = np.clip((values - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(v8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = lo + t / 255 * (hi - lo)
    above = values > thr
    w1 = above.mean()
    if w1 in (0, 1):
        return thr, 0.0
    between = w1 * (1 - w1) * (values[above].mean() - values[~above].mean()) ** 2
    return thr, float(between / values.var())


def detect_electrodes(photo, feature="auto", sigma=15, n=2, min_area_frac=0.02):
    """
    Find the electrodes in ``photo`` (BGR).

    feature : "warm" (red minus blue; e.g. metal pads that look reddish under
        dark field), "bright", "dark", or "auto" (the one that best splits the
        sample into two classes).
    sigma : smoothing (px of the photo) before thresholding; larger than the
        spacing between wires.

    Returns (masks, info): ``masks`` is a list of ``n`` boolean arrays ordered
    left -> right (or top -> bottom if they are stacked vertically).
    """
    photo = np.asarray(photo)
    if photo.ndim == 2:
        photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
    feats = ("warm", "bright", "dark") if feature == "auto" else (feature,)
    _, bright = _feature_map(photo, "bright", sigma)
    sample = bright > max(np.percentile(bright, 5) * 1.5, 1.0)   # exclude black outside the sample
    if sample.mean() < 0.2:
        sample[:] = True
    scored = []
    for ft in feats:
        fmap, _ = _feature_map(photo, ft, sigma)
        thr, sep = _otsu(fmap[sample])
        scored.append((sep, ft, fmap, thr))
    sep, ft, fmap, thr = max(scored, key=lambda x: x[0])

    mask = (fmap > thr) & sample
    k = int(2 * round(sigma) + 1)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    mask = ndimage.binary_fill_holes(mask)
    lab, nlab = ndimage.label(mask)
    if nlab < n:
        raise RuntimeError(f"Found {nlab} electrode region(s) with feature '{ft}', need {n}. "
                           "Try another feature or a different smoothing.")
    sizes = ndimage.sum(mask, lab, range(1, nlab + 1))
    order = np.argsort(sizes)[::-1][:n] + 1
    if sizes[order[-1] - 1] < min_area_frac * mask.size:
        raise RuntimeError(f"Only {sum(s >= min_area_frac * mask.size for s in sizes)} large "
                           f"region(s) found with feature '{ft}'. Try another feature.")
    cents = [ndimage.center_of_mass(lab == k) for k in order]
    ys, xs = [c[0] for c in cents], [c[1] for c in cents]
    axis = 1 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 0
    order = [k for _, k in sorted(zip([c[axis] for c in cents], order))]
    masks = [lab == k for k in order]
    return masks, dict(feature=ft, separability=sep,
                       layout="left-right" if axis == 1 else "top-bottom",
                       scores={s[1]: round(s[0], 3) for s in scored})


# =============================================================================
#  3. Transfer to the image
# =============================================================================
def warp_mask(mask, M, shape):
    """Warp a photo mask into the image (nearest neighbor), as a boolean array."""
    h, w = shape[:2]
    return cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0


def warp_photo(photo, M, shape):
    """The photo resampled into the image coordinates (for visual checks)."""
    h, w = shape[:2]
    return cv2.warpAffine(photo, M, (w, h), flags=cv2.INTER_LINEAR)


def photo_footprint(photo_shape, M):
    """Corners of the photo in image coordinates, as an (4, 2) array (x, y)."""
    h, w = photo_shape[:2]
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.transform(c, M).reshape(-1, 2)


def electrodes_from_photo(image, photo, scale=None, feature="auto", sigma=15, verbose=False):
    """
    Register ``photo`` on ``image`` and return the electrode masks in image
    coordinates: (mask1, mask2, info). mask1 is the left (or top) electrode,
    used as Vin; mask2 the right (or bottom) one, Vout.

    ``info`` holds the registration results (see ``register``), the detection
    results (see ``detect_electrodes``), ``M`` and a list of ``warnings``.
    """
    reg = register(photo, image, scale=scale, verbose=verbose)
    photo_masks, det = detect_electrodes(photo, feature=feature, sigma=sigma)
    masks = [warp_mask(m, reg["M"], image.shape) for m in photo_masks]
    warnings = []
    margin = 3
    for name, m in zip(("Vin", "Vout"), photo_masks):
        if m[:margin].any() or m[-margin:].any() or m[:, :margin].any() or m[:, -margin:].any():
            warnings.append(f"The {name} electrode touches the border of the photo: it may "
                            "continue outside the photo, or the detection may include some "
                            "background there. Check that edge.")
    if reg["residual_median"] > 3:
        warnings.append(f"Registration residual is high ({reg['residual_median']:.1f} px): "
                        "check the alignment visually.")
    info = dict(reg, **{f"detect_{k}": v for k, v in det.items()}, warnings=warnings)
    return masks[0], masks[1], info


# =============================================================================
#  Visual check
# =============================================================================
def check_image(image, photo, info, masks, max_size=1600):
    """
    BGR image to verify the result: left, the image with the electrodes and the
    outline of the photo; right, a zoom between the electrodes with the image in
    magenta and the aligned photo in green (wires present in both look white).
    """
    M = info["M"]
    img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = img[..., :3].copy()
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = img.shape[:2]
    over = img.copy()
    for m, color in zip(masks, ((0, 0, 255), (255, 0, 0))):
        over[m] = (0.6 * over[m] + 0.4 * np.array(color)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cnts, -1, color, max(2, w // 600))
    cv2.polylines(over, [photo_footprint(photo.shape, M).astype(np.int32)], True,
                  (0, 255, 255), max(2, w // 600))
    s = max_size / max(h, w)
    left = cv2.resize(over, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    # zoom: midpoint between the two electrodes, inside the photo
    cy = int(np.mean([np.argwhere(m)[:, 0].mean() for m in masks]))
    cx = int(np.mean([np.argwhere(m)[:, 1].mean() for m in masks]))
    half = max(100, min(h, w) // 12)
    y0, x0 = max(0, cy - half), max(0, cx - half)
    sl = (slice(y0, min(h, cy + half)), slice(x0, min(w, cx + half)))
    clahe = cv2.createCLAHE(3.0, (8, 8))
    a = clahe.apply(_gray(img)[sl])
    b = clahe.apply(_gray(warp_photo(photo, M, img.shape))[sl])
    zoom = np.dstack([a, b, a])
    zoom = cv2.resize(zoom, (left.shape[0], left.shape[0]), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(left, (int(x0 * s), int(y0 * s)),
                  (int((x0 + 2 * half) * s), int((y0 + 2 * half) * s)), (255, 0, 255), 2)
    text = (f"scale {info['scale']:.3f}  rotation {info['rotation_deg']:.2f} deg  "
            f"{info['inliers']} matches  residual {info['residual_median']:.2f} px")
    cv2.putText(left, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(zoom, "zoom: magenta = image, green = photo, white = both",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return np.hstack([left, zoom])


# =============================================================================
#  Command line
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(
        description="Find the electrodes of a later photo (other magnification/rotation) "
                    "on an existing image of the network, and save them as electrode masks.")
    p.add_argument("image", help="Image of the network (e.g. the stitched mosaic).")
    p.add_argument("photo", help="Later photo that shows the electrodes.")
    p.add_argument("-o", "--output", default=None,
                   help="Base name for the outputs (default: the image path without extension). "
                        "Writes <base>_electrode_mask1.tif, _mask2.tif and _electrode_check.jpg.")
    p.add_argument("--scale", type=float, default=None,
                   help="Approximate image px per photo px (e.g. 2 for 5x on 10x). Default: search.")
    p.add_argument("--feature", choices=FEATURES, default="auto",
                   help="How the electrodes differ from the bare network in the photo.")
    p.add_argument("--sigma", type=float, default=15, help="Smoothing for the detection (photo px).")
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    verbose = not args.quiet
    image = imread(args.image, cv2.IMREAD_COLOR)
    photo = imread(args.photo, cv2.IMREAD_COLOR)
    m1, m2, info = electrodes_from_photo(image, photo, scale=args.scale, feature=args.feature,
                                         sigma=args.sigma, verbose=verbose)
    base = args.output or os.path.splitext(args.image)[0]
    imwrite(base + "_electrode_mask1.tif", m1.astype(np.uint8) * 255)
    imwrite(base + "_electrode_mask2.tif", m2.astype(np.uint8) * 255)
    imwrite(base + "_electrode_check.jpg", check_image(image, photo, info, (m1, m2)),
            (cv2.IMWRITE_JPEG_QUALITY, 90))
    if verbose:
        print(f"Aligned: scale {info['scale']:.3f}, rotation {info['rotation_deg']:.2f} deg, "
              f"{info['inliers']}/{info['matches']} consistent matches, "
              f"residual {info['residual_median']:.2f} px (90%: {info['residual_p90']:.2f}).")
        print(f"Electrodes detected with feature '{info['detect_feature']}' "
              f"({info['detect_layout']}).")
        for wmsg in info["warnings"]:
            print("[warning]", wmsg)
        print(f"Saved {base}_electrode_mask1.tif (Vin), _electrode_mask2.tif (Vout) "
              f"and _electrode_check.jpg.")


if __name__ == "__main__":
    main()

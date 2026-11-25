"""
WireTracing
===========

Extraction of the connection graph of a nanowire network **directly from the
grayscale photomicrograph**, without binarizing or skeletonizing it. Each node
of the resulting graph is a whole wire (not a wire segment between two
junctions), and each edge is a contact between two wires.

How it works
------------
1. **Orientation scores.** The image is filtered with a bank of elongated
   ridge filters (second derivative of an anisotropic Gaussian) at
   ``n_orientations`` angles. ``R[k, y, x]`` is the "wire-ness" of pixel
   (x, y) along orientation k, in units of wire contrast (gray levels above
   the local background). Where two wires cross, each one keeps responding in
   its own orientation layer, which is the information a binary image and its
   skeleton throw away.
2. **Tracing.** Starting from the strongest ridge points, each wire is
   followed in both directions, step by step, staying in the orientation
   layers close to its current direction. At a crossing the other wire lives
   in a different layer, so the tracer goes straight through instead of
   turning or stopping. Short dim stretches (``max_gap``) are bridged.
3. **Joining.** Pieces of the same wire that the tracer could not connect
   (shallow-angle crossings, bright clumps, dim spots) are joined when their
   ends are close and their directions continue smoothly.
4. **Contacts.** Two wires are in contact when their center lines come
   within ``contact_distance`` of each other (crossings, side contacts) or an
   end of one is within ``end_tolerance`` of the other (a wire ending on
   another). Wires touching an electrode mask get ``Vin_*`` / ``Vout_*``
   neighbours, as in ``SkeletonAnalysis``.

Result
------
* ``WA.wires``: list of polylines, one ``(n, 2)`` array of (x, y) per wire.
* ``WA.L``: graph whose nodes are wires (attributes ``x``, ``y`` centroid,
  ``length``, endpoints ``x0, y0, x1, y1`` and ``path``, the polyline as
  ``"x,y;x,y;..."``) and edges are contacts (attributes ``x``, ``y``).
  Electrode nodes are called ``Vin_i`` / ``Vout_i`` as before.
* ``WA.G``: spatial graph equivalent to ``SkeletonAnalysis.G`` (nodes =
  contacts, edges = wire pieces between consecutive contacts), for
  comparison and for the existing plotting functions.
"""

import math
from collections import OrderedDict

import numpy as np
import cv2
import networkx as nx
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree

# Reuse the loaders and the scikit-image compatibility helpers
from SkeletonAnalysis import _as_gray_uint8, _load_mask, remove_small_objects


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
def ridge_kernel(theta, sigma_across, sigma_along):
    """
    Elongated ridge detector oriented along ``theta`` (radians, x right, y
    down): minus the second derivative across the wire of an anisotropic
    Gaussian. Zero mean, so a smooth background gives no response.
    """
    half = int(np.ceil(3 * max(sigma_across, sigma_along)))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    c, s = np.cos(theta), np.sin(theta)
    v = x * c + y * s            # along the wire
    u = -x * s + y * c           # across the wire
    g = np.exp(-u ** 2 / (2 * sigma_across ** 2) - v ** 2 / (2 * sigma_along ** 2))
    k = (1 - u ** 2 / sigma_across ** 2) * g
    k -= k.mean()
    return k.astype(np.float32)


def _line_response_scale(width, sigma_across, sigma_along, blur=0.7):
    """Response of the kernel to an ideal straight wire of contrast 1."""
    size = int(8 * max(sigma_across, sigma_along, width)) | 1
    img = np.zeros((size, size), np.float32)
    c = size // 2
    ss = 8
    big = np.zeros((size * ss, size * ss), np.uint8)
    cv2.line(big, (0, c * ss + ss // 2), (size * ss - 1, c * ss + ss // 2), 255,
             thickness=max(int(round(width * ss)), 1))
    img = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    img = cv2.GaussianBlur(img, (0, 0), blur)
    k = ridge_kernel(0.0, sigma_across, sigma_along)
    r = cv2.filter2D(img, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    return float(r[c - 1:c + 2, c].max())


def estimate_noise(img):
    """Standard deviation of the pixel noise (Immerkaer, 1996)."""
    img = img.astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(img, cv2.CV_32F, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6)


def estimate_wire_width(img, n=3000, seed=0):
    """
    Typical wire width in px (bright wires on a darker background), measured
    as the full width at half maximum of the intensity profile across the
    strongest ridges, corrected for the blur of a typical optical system.
    Returns None if no clear wires are found.
    """
    from scipy.ndimage import map_coordinates
    img = np.asarray(img)
    if len(np.unique(img[::7, ::7])) <= 2:
        # binary image: twice the distance to the background along the skeleton
        from skimage.morphology import skeletonize
        fg = img > img.min()
        dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
        d = dist[skeletonize(fg)]
        return float(2 * np.median(d) - 1) if d.size else None
    img = img.astype(np.float32)
    g = cv2.GaussianBlur(img, (0, 0), 1.5)
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    lam = ((dxx + dyy) - np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)) / 2
    resp = -lam
    peaks = (resp == cv2.dilate(resp, np.ones((5, 5), np.uint8))) & (resp > np.percentile(resp, 99))
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return None
    if len(ys) > n:
        sel = np.random.default_rng(seed).choice(len(ys), n, replace=False)
        ys, xs = ys[sel], xs[sel]
    # direction across the ridge: eigenvector of the most negative eigenvalue
    a, b, l = dxx[ys, xs], dxy[ys, xs], lam[ys, xs]
    vx, vy = b.copy(), l - a
    nrm = np.hypot(vx, vy)
    flat = nrm < 1e-6
    vx[flat], vy[flat], nrm[flat] = 1, 0, 1
    vx, vy = vx / nrm, vy / nrm
    t = np.arange(-10, 10.01, 0.25)
    X = xs[:, None] + t[None, :] * vx[:, None]
    Y = ys[:, None] + t[None, :] * vy[:, None]
    prof = map_coordinates(img, [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(X.shape)
    c0 = len(t) // 2
    widths = []
    for p in prof:
        i = c0 - 2 + int(np.argmax(p[c0 - 2:c0 + 3]))
        peak = p[i]
        bgl = np.median(np.sort(p[:i])[:8]) if i > 8 else p[0]
        bgr = np.median(np.sort(p[i + 1:])[:8]) if len(p) - i > 9 else p[-1]
        if peak - max(bgl, bgr) < 10:
            continue
        hl, hr = (peak + bgl) / 2, (peak + bgr) / 2
        j = i
        while j > 0 and p[j] > hl:
            j -= 1
        if j == 0:
            continue
        xl = t[j] + (hl - p[j]) / (p[j + 1] - p[j]) * 0.25
        j = i
        while j < len(p) - 1 and p[j] > hr:
            j += 1
        if j == len(p) - 1:
            continue
        xr = t[j - 1] + (p[j - 1] - hr) / (p[j - 1] - p[j]) * 0.25
        widths.append(xr - xl)
    if not widths:
        return None
    fwhm = float(np.median(widths))
    return float(np.sqrt(max(fwhm ** 2 - 3.5, 1.0)))


def line_opening(img, length, n_orientations=16):
    """
    Maximum over ``n_orientations`` of the grayscale opening of ``img`` with a
    straight segment of ``length`` px. A pixel keeps its brightness only if a
    segment that long fits through bright pixels: wires survive (they are
    continuous along their direction), isolated spots and rows of spots with
    darker gaps between them do not.
    """
    img = np.asarray(img)
    L = int(round(length)) | 1
    c = L // 2
    out = np.zeros_like(img)
    for th in np.arange(n_orientations) * np.pi / n_orientations:
        k = np.zeros((L, L), np.uint8)
        dx, dy = np.cos(th) * c, np.sin(th) * c
        cv2.line(k, (int(round(c - dx)), int(round(c - dy))),
                 (int(round(c + dx)), int(round(c + dy))), 1, 1)
        np.maximum(out, cv2.morphologyEx(img, cv2.MORPH_OPEN, k), out=out)
    return out


def empty_margin_mask(img, grow=5, min_area=2000):
    """
    Empty areas of a stitched mosaic: regions that are exactly 0, touch the
    image border and cover at least ``min_area`` px, grown by ``grow`` px.
    Their straight edges would otherwise be traced as wires. None if there
    are none.
    """
    zero = (np.asarray(img) == 0).astype(np.uint8)
    if not zero.any():
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(zero, connectivity=8)
    H, W = zero.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area >= min_area and (x == 0 or y == 0 or x + w == W or y + h == H):
            keep[i] = True
    if not keep.any():
        return None
    mask = keep[lab].astype(np.uint8)
    if grow > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    return mask.astype(bool)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def resample_polyline(pts, step=1.0):
    """Points along the polyline every ``step`` px (first and last included)."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts.copy()
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if d[-1] == 0:
        return pts[:1].copy()
    s = np.r_[np.arange(0, d[-1], step), d[-1]]
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


def smooth_polyline(pts, window=5):
    """Moving average of the points (ends kept fixed), removes tracing jitter."""
    pts = np.asarray(pts, float)
    if window < 2 or len(pts) <= window:
        return pts
    k = np.ones(window) / window
    pad = window // 2
    ext = np.vstack([np.repeat(pts[:1], pad, 0), pts, np.repeat(pts[-1:], pad, 0)])
    out = np.c_[np.convolve(ext[:, 0], k, "valid"), np.convolve(ext[:, 1], k, "valid")]
    out[0], out[-1] = pts[0], pts[-1]
    return out


def polyline_length(pts):
    pts = np.asarray(pts, float)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def _end_tangent(pts, end, span=6.0):
    """Unit vector pointing *outwards* from one end of the polyline."""
    pts = np.asarray(pts, float)
    if end == 0:
        a, rest = pts[0], pts[1:]
    else:
        a, rest = pts[-1], pts[-2::-1]
    if len(rest) == 0:
        return np.array([1.0, 0.0])
    d = np.hypot(*(rest - a).T)
    i = int(np.searchsorted(d, span)) if d[-1] >= span else len(rest) - 1
    i = min(i, len(rest) - 1)
    v = a - rest[i]
    n = np.hypot(*v)
    return v / n if n > 0 else np.array([1.0, 0.0])


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
class WireAnalysis:
    """
    Graph of whole wires from a grayscale micrograph.

    Parameters
    ----------
    img_path, img : path to the micrograph, or the image as an array (gray or
        RGB, any dtype). Wires must be brighter than the background (optical
        dark field, SEM); for bright-field images pass ``invert=True``.
    wire_width : approximate width of a wire in pixels; estimated from the
        image when None.
    """

    def __init__(self, img_path=None, img=None, wire_width=None, invert=False):
        if img_path is None and img is None:
            raise ValueError("Pass img_path or img.")
        self.img_path = img_path
        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Could not read the image: {img_path}")
            if img.ndim == 3:
                img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        self.img = _as_gray_uint8(img)
        if invert:
            self.img = 255 - self.img
        # A binary image has flat-topped wires with sharp edges, which give two
        # ridges (one per edge). Blurring it turns each wire into one ridge.
        self.is_binary = len(np.unique(self.img[::7, ::7])) <= 2
        self.img_original = self.img
        self.original_img_path = img_path
        if wire_width is None:
            wire_width = estimate_wire_width(self.img) or 3.0
        self.wire_width = float(wire_width)
        if self.is_binary:
            self.img = cv2.GaussianBlur(self.img, (0, 0), max(0.7, 0.4 * self.wire_width))
        self.has_electrodes = False
        self.exclude_mask = None
        self.removed_wires = set()    # wires deleted by hand (indices into self.wires)
        self.progress = None          # callback(stage, fraction 0-1 overall, detail)
        self._stage = None
        self._last_report = 0.0
        self.wires = []

    # --------------------------------------------------------------- progress
    # Share of the total time of each stage (measured on M1/M3/M4 sized images)
    STAGES = [("Filtering the image", 0.15), ("Tracing wires", 0.63),
              ("Joining pieces of wires", 0.05), ("Closing gaps at wire ends", 0.04),
              ("Finding contacts", 0.05), ("Measuring wires", 0.06),
              ("Building the graph", 0.02)]

    def _report(self, stage, frac=0.0, detail="", force=False):
        """Send progress to ``self.progress`` (at most ~4 times per second)."""
        if self.progress is None:
            return
        import time
        now = time.time()
        if not force and stage == self._stage and now - self._last_report < 0.25:
            return
        self._stage, self._last_report = stage, now
        done, share = 0.0, 0.0
        for name, w in self.STAGES:
            if name == stage:
                share = w
                break
            done += w
        try:
            self.progress(stage, min(1.0, done + share * min(max(frac, 0.0), 1.0)), detail)
        except Exception:
            pass                        # a broken display must not stop the analysis

    # ------------------------------------------------------------------ input
    def load_electrodes(self, path1, path2):
        """Load the Vin and Vout masks, given as paths or as arrays."""
        m1 = binary_fill_holes(_load_mask(path1) > 0)
        m2 = binary_fill_holes(_load_mask(path2) > 0)
        self.electrode_mask1 = remove_small_objects(m1, max_size=500)
        self.electrode_mask2 = remove_small_objects(m2, max_size=500)
        self.has_electrodes = True

    def load_exclusion(self, mask):
        """
        Areas to ignore (path or array; nonzero = ignore), e.g. a region of
        dust or substrate outside the network that is still traced. No wire is
        traced inside them.
        """
        m = _load_mask(mask) > 0
        if m.shape != self.img.shape[:2]:
            raise ValueError(f"The exclusion mask is {m.shape}, the image {self.img.shape[:2]}.")
        self.exclude_mask = m if self.exclude_mask is None else (self.exclude_mask | m)

    # ------------------------------------------------------ 1. orientation scores
    def orientation_scores(self, n_orientations=16, sigma_across=None, sigma_along=None,
                           tile_size=256, max_cache_mb=1024, continuity_length=None):
        """
        Set up the orientation scores: the ridge response of every pixel along
        each of ``n_orientations`` directions, in gray levels of wire contrast.

        They are computed by tiles of ``tile_size`` px on demand and kept in a
        cache of at most ``max_cache_mb`` MB, so memory does not grow with
        n_orientations x image size (a 450 Mpx mosaic would need ~14 GB
        otherwise). This pass also stores, per pixel, the best response
        (``self.R_max``) and its orientation (``self.R_arg``), used for seeds.
        """
        w = self.wire_width
        sa = sigma_across or max(0.8, 0.35 * w)
        sl = sigma_along or max(3.0, 1.3 * w)
        self.sigma_across, self.sigma_along = sa, sl
        # Continuity: how much of each pixel's contrast survives an opening by
        # line segments of length L (~1 on a wire, ~0 on dots and rows of dots
        # with gaps). Used to judge seeds and traces, not to filter the image.
        L = continuity_length if continuity_length is not None else 5 * w
        self.continuity_length = L
        self._src = self.img
        if L and L >= 3:
            r_bg = int(round(1.5 * w)) + 1
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_bg + 1,) * 2)
            self._bg = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, disk)
            self._cont = line_opening(self.img, L, n_orientations)
        else:
            self._bg = self._cont = None
        K = int(n_orientations)
        if K > 32:
            raise ValueError("n_orientations must be <= 32.")
        self.n_orientations = K
        self.thetas = np.arange(K) * np.pi / K
        scale = _line_response_scale(w, sa, sl)
        self._kernels = [ridge_kernel(th, sa, sl) / scale for th in self.thetas]
        self._pad = self._kernels[0].shape[0] // 2
        H, W = self.img.shape
        # Whole image as a single tile when it fits in the cache (fastest)
        if K * (H + 1) * (W + 1) * 2 <= max_cache_mb * 2 ** 20:
            tile_size = max(H, W)
        self._tile = int(tile_size)
        tile_bytes = K * (min(self._tile, H) + 1) * (min(self._tile, W) + 1) * 2
        self._cache_max = max(4, int(max_cache_mb * 2 ** 20 // tile_bytes))
        self._cache = OrderedDict()
        inside = (self.electrode_mask1 | self.electrode_mask2) if self.has_electrodes else None
        if self.exclude_mask is not None:
            inside = self.exclude_mask if inside is None else (inside | self.exclude_mask)
        margin = empty_margin_mask(self.img, int(round(2 * sa + w)))
        self.margin_mask = margin
        if margin is not None:
            inside = margin if inside is None else (inside | margin)
        self._inside = inside

        H, W = self.img.shape
        T = self._tile
        self.R_max = np.zeros((H, W), np.float16)
        self.R_arg = np.zeros((H, W), np.uint8)
        self.R_perp = np.zeros((H, W), np.float16)
        n_ty, n_tx = (H + T - 1) // T, (W + T - 1) // T
        self._report("Filtering the image", 0.0, f"{K} orientations", force=True)
        for ty in range(n_ty):
            for tx in range(n_tx):
                self._setup_block = (ty * n_tx + tx, n_ty * n_tx)
                R = self._get_tile(ty, tx)
                h, w_ = min(T, H - ty * T), min(T, W - tx * T)
                r = R[:, :h, :w_]
                arg = r.argmax(0)
                sl_ = (slice(ty * T, ty * T + h), slice(tx * T, tx * T + w_))
                self.R_arg[sl_] = arg
                self.R_max[sl_] = r.max(0)
                # response across the best orientation: ~0 on a wire, as large
                # as R_max on a round spot (dust, a particle, a blurred dot)
                perp = ((arg.astype(np.intp) + K // 2) % K)[None]
                self.R_perp[sl_] = np.take_along_axis(r, perp, 0)[0]
        self._setup_block = None
        # Noise of the response, in the same units (for automatic thresholds)
        knorm = float(np.sqrt((self._kernels[0] ** 2).sum()))
        self.noise = estimate_noise(self.img) * knorm

    def _get_tile(self, ty, tx):
        """Orientation scores of one tile (K x (T+1) x (T+1), float16), cached."""
        key = (ty, tx)
        R = self._cache.get(key)
        if R is not None:
            self._cache.move_to_end(key)
            return R
        H, W = self.img.shape
        T, P = self._tile, self._pad
        y0, x0 = ty * T, tx * T
        y1, x1 = min(y0 + T + 1, H), min(x0 + T + 1, W)
        ya, yb = max(y0 - P, 0), min(y1 + P, H)
        xa, xb = max(x0 - P, 0), min(x1 + P, W)
        crop = self._src[ya:yb, xa:xb].astype(np.float32)
        top, left = P - (y0 - ya), P - (x0 - xa)
        bottom, right = P - (yb - y1), P - (xb - x1)
        if top or left or bottom or right:
            crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REFLECT)
        h, w_ = y1 - y0, x1 - x0
        R = np.zeros((self.n_orientations, h + 1, w_ + 1), np.float16)
        blk = getattr(self, "_setup_block", None)
        for k, ker in enumerate(self._kernels):
            if blk is not None:
                b, nb = blk
                self._report("Filtering the image", (b + k / len(self._kernels)) / nb,
                             f"orientation {k + 1} of {len(self._kernels)}"
                             + (f", block {b + 1} of {nb}" if nb > 1 else ""))
            r = cv2.filter2D(crop, cv2.CV_32F, ker, borderType=cv2.BORDER_CONSTANT)
            R[k, :h, :w_] = np.maximum(r[P:P + h, P:P + w_], 0)
        if self._inside is not None:
            R[:, :h, :w_][:, self._inside[y0:y1, x0:x1]] = 0
        self._cache[key] = R
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return R

    # ---------------------------------------------------------------- sampling
    def _sample(self, k, xs, ys):
        """Bilinear sample of the scores of layer(s) ``k`` at float positions."""
        H, W = self.img.shape
        T = self._tile
        xs = np.asarray(xs, np.float64)
        ys = np.asarray(ys, np.float64)
        ok = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
        x = np.clip(xs, 0, W - 1.001)
        y = np.clip(ys, 0, H - 1.001)
        x0 = x.astype(np.intp)
        y0 = y.astype(np.intp)
        fx = x - x0
        fy = y - y0
        k = np.broadcast_to(np.asarray(k, np.intp) % self.n_orientations, x0.shape)
        tx, ty = x0 // T, y0 // T
        out = np.empty(x0.shape, np.float64)
        if (tx == tx.flat[0]).all() and (ty == ty.flat[0]).all():
            groups = [(int(ty.flat[0]), int(tx.flat[0]), slice(None))]
        else:
            keys = ty * (W // T + 2) + tx
            groups = [(int(ty[keys == u][0]), int(tx[keys == u][0]), keys == u)
                      for u in np.unique(keys)]
        for gy, gx, sel in groups:
            R = self._get_tile(gy, gx)
            lx, ly = x0[sel] - gx * T, y0[sel] - gy * T
            kk, ax, ay = k[sel], fx[sel], fy[sel]
            out[sel] = (R[kk, ly, lx] * (1 - ax) * (1 - ay) + R[kk, ly, lx + 1] * ax * (1 - ay)
                        + R[kk, ly + 1, lx] * (1 - ax) * ay + R[kk, ly + 1, lx + 1] * ax * ay)
        return np.where(ok, out, 0.0)

    def _sample_line(self, k, q, n, offs):
        """Scores of layer ``k`` at the points ``q + offs * n`` (fast path)."""
        H, W = self.img.shape
        T = self._tile
        xs = q[0] + offs * n[0]
        ys = q[1] + offs * n[1]
        xa, xb, ya, yb = xs.min(), xs.max(), ys.min(), ys.max()
        if xa < 0 or ya < 0 or xb >= W - 1 or yb >= H - 1:
            return self._sample(k, xs, ys)
        tx, ty = int(xa) // T, int(ya) // T
        if int(xb) // T != tx or int(yb) // T != ty:
            return self._sample(k, xs, ys)
        Rk = self._get_tile(ty, tx)[k % self.n_orientations]
        xs = xs - tx * T
        ys = ys - ty * T
        x0 = xs.astype(np.intp)
        y0 = ys.astype(np.intp)
        fx = xs - x0
        fy = ys - y0
        return ((Rk[y0, x0] * (1 - fx) + Rk[y0, x0 + 1] * fx) * (1 - fy)
                + (Rk[y0 + 1, x0] * (1 - fx) + Rk[y0 + 1, x0 + 1] * fx) * fy)

    def _sample_layers(self, ks, x, y):
        """Scores of the layers ``ks`` at one point (fast path)."""
        H, W = self.img.shape
        if not (0 <= x < W - 1 and 0 <= y < H - 1):
            return self._sample(ks, np.full(len(ks), x), np.full(len(ks), y))
        T = self._tile
        xi, yi = int(x), int(y)
        tx, ty = xi // T, yi // T
        R = self._get_tile(ty, tx)
        lx, ly = xi - tx * T, yi - ty * T
        fx, fy = x - xi, y - yi
        ks = ks % self.n_orientations
        blk = R[ks, ly:ly + 2, lx:lx + 2].astype(np.float64)
        return ((blk[:, 0, 0] * (1 - fx) + blk[:, 0, 1] * fx) * (1 - fy)
                + (blk[:, 1, 0] * (1 - fx) + blk[:, 1, 1] * fx) * fy)

    def _occupied(self, k, x, y):
        return bool((int(self.occupancy[y, x]) >> (k % self.n_orientations)) & 1)

    def _bin(self, phi):
        """Orientation index of a heading angle (any direction)."""
        K = self.n_orientations
        return int(round((phi % np.pi) * K / np.pi)) % K

    # --------------------------------------------------------------- 2. tracing
    def _probe(self, p, phi, dist, offs, window, low):
        """
        Look ``dist`` px ahead of ``p`` along heading ``phi``: re-center on the
        ridge (lateral offsets ``offs``) and measure the local orientation among
        the layers within ``window`` of ``phi``. Returns (point, signed
        orientation difference w.r.t. phi, response).
        """
        dth = np.pi / self.n_orientations
        c, s_ = math.cos(phi), math.sin(phi)
        n = (-s_, c)
        q = np.array([p[0] + c * dist, p[1] + s_ * dist])
        kb = self._bin(phi)
        r = self._sample_line(kb, q, n, offs) - self._offs_penalty * low
        o = offs[int(np.argmax(r))]
        q = np.array([q[0] + o * n[0], q[1] + o * n[1]])
        nw = 2 * window + 1
        rk = self._sample_layers(kb + np.arange(-window, window + 1), q[0], q[1])
        j = int(np.argmax(rk))
        frac = 0.0
        if 0 < j < nw - 1:
            a, b, c = rk[j - 1], rk[j], rk[j + 1]
            den = a - 2 * b + c
            if den < 0:
                frac = min(max(0.5 * float(a - c) / float(den), -0.5), 0.5)
        th_meas = (kb + (j - window) + frac) * dth
        diff = (th_meas - phi + np.pi / 2) % np.pi - np.pi / 2
        return q, diff, float(rk[j])

    def _follow(self, p, phi, low, step, max_turn, max_gap, search, max_steps,
                window=1, memory=None, recovery=2, pass_length=None):
        """
        Follow a ridge from point ``p`` with heading ``phi`` (radians, direction
        of travel). Returns the list of visited points (excluding ``p``).

        At each step the tracer moves ``step`` px ahead, re-centers itself on
        the ridge (at most ``search`` px sideways) and measures the local
        orientation among the layers within ``window`` of its heading. The
        heading follows that measurement, but turns at most ``max_turn`` per
        step relative to the direction of the last ``memory`` px of the
        trace, so a crossing wire cannot pull the tracer off its own wire.

        When the ridge is lost straight ahead the tracer first tries sharper
        turns (up to ``recovery`` extra multiples of ``max_turn``, for tightly
        bent wires) and then jumps over gaps of up to ``max_gap`` steps.
        """
        offs = np.arange(-search, search + 1e-9, 0.25)
        self._offs_penalty = 0.3 * np.abs(offs)
        H, W = self.img.shape
        memory = memory or 6 * self.wire_width
        n_mem = max(1, int(round(memory / step)))
        pts = []
        hist = [np.asarray(p, float)]
        p = hist[0]
        turns = [m * s_ for m in range(1, recovery + 1) for s_ in (1, -1)]
        pass_steps = int(round(pass_length / step)) if pass_length else 0
        run_start = None
        for _ in range(max_steps):
            # heading prior: direction of the recent trace (momentum)
            if len(hist) > n_mem // 2 + 1:
                v = hist[-1] - hist[-1 - min(n_mem, len(hist) - 1)]
                prior = math.atan2(v[1], v[0])
                dphi = (phi - prior + np.pi) % (2 * np.pi) - np.pi
                phi = prior + min(max(dphi, -max_turn), max_turn)
            q, diff, resp = self._probe(p, phi, step, offs, window, low)
            jump, phi_used = 1, phi
            if resp < low and recovery:
                best = None
                for m in turns:
                    ph = phi + m * max_turn
                    q2, d2, r2 = self._probe(p, ph, step, offs, window, low)
                    # the orientation found must agree with the new heading
                    if r2 >= low and abs(d2) <= max_turn and (best is None or r2 > best[2]):
                        best = (q2, d2, r2, ph)
                if best is not None:
                    q, diff, resp, phi_used = best
            if resp < low:
                for jump in range(2, max_gap + 2):
                    q, diff, resp = self._probe(p, phi, step * jump, offs, window, low)
                    if resp >= low:
                        break
            if resp < low or not (0 <= q[0] <= W - 1 and 0 <= q[1] <= H - 1):
                return pts if run_start is None else pts[:run_start]
            if jump > 1:                            # fill the gap linearly
                for t in range(1, jump):
                    pts.append(p + (q - p) * t / jump)
            # Running along a wire already traced in this direction: either a
            # shallow-angle crossing (pass through it) or the same wire traced
            # twice (after ``pass_steps`` steps, drop the overlap and stop).
            xi, yi = int(round(q[0])), int(round(q[1]))
            if self._occupied(self._bin(phi_used + diff), xi, yi):
                if run_start is None:
                    run_start = len(pts)
                if len(pts) + 1 - run_start > pass_steps:
                    return pts[:run_start]
            else:
                run_start = None
            recovered = phi_used != phi
            phi = phi_used + min(max(diff, -max_turn), max_turn)
            p = q
            pts.append(p.copy())
            hist.append(p)
            if recovered:
                # after a sharp turn, the momentum restarts from the new heading
                hist = hist[-2:]
        return pts

    def trace(self, seed_threshold=None, low_threshold=None, step=1.5, max_turn_deg=4,
              max_gap=2, search=0.5, window=1, memory=None, recovery=2, smooth=5, pass_length=None,
              min_length=None, max_length=20000, max_isotropy=0.5, min_continuity=0.6):
        """
        Trace every wire. Thresholds are in gray levels of wire contrast;
        by default they adapt to the noise of the image (seed: max(10, 6 noise),
        continuation: max(5, 3 noise)). ``max_turn_deg`` (per step of
        ``step`` px) limits the curvature a wire can have: 4 deg per 1.5 px is
        a radius of ~21 px.
        """
        if not hasattr(self, "R_max"):
            self.orientation_scores()
        w = self.wire_width
        high = seed_threshold if seed_threshold is not None else max(10.0, 6 * self.noise)
        low = low_threshold if low_threshold is not None else max(5.0, 3 * self.noise)
        self.seed_threshold, self.low_threshold = high, low
        min_length = min_length if min_length is not None else 4 * w
        K = self.n_orientations
        H, W = self.img.shape
        # one bit per orientation layer: "a wire already runs here, this way"
        self.occupancy = np.zeros((H, W), np.uint16 if K <= 16 else np.uint32)
        occ_r = max(1, int(round(0.6 * w)))

        # seeds: local maxima of the best response above the seed threshold
        Rmax = self.R_max
        Rperp = self.R_perp
        ys, xs, vs = [], [], []
        band = 512
        k3 = np.ones((3, 3), np.uint8)
        for r0 in range(0, H, band):
            a, b = max(r0 - 1, 0), min(r0 + band + 1, H)
            blk = Rmax[a:b].astype(np.float32)
            pk = (blk >= high) & (blk == cv2.dilate(blk, k3))
            pk &= Rperp[a:b].astype(np.float32) <= max_isotropy * blk
            if self._cont is not None:
                bgb = self._bg[a:b].astype(np.float32)
                cb = self.img[a:b].astype(np.float32) - bgb
                ob = self._cont[a:b].astype(np.float32) - bgb
                pk &= ob >= min_continuity * cb
            pk[:r0 - a] = False
            pk[min(r0 + band, H) - a:] = False
            y, x = np.nonzero(pk)
            ys.append(y + a)
            xs.append(x)
            vs.append(blk[y, x])
        ys, xs, vs = np.concatenate(ys), np.concatenate(xs), np.concatenate(vs)
        order = np.argsort(-vs, kind="stable")
        max_steps = int(max_length / step)
        max_turn = np.deg2rad(max_turn_deg)
        wires = []
        n_seeds = len(order)
        self._report("Tracing wires", 0.0, f"{n_seeds} starting points", force=True)
        for n_done, idx in enumerate(order):
            if n_done % 200 == 0:
                self._report("Tracing wires", n_done / max(n_seeds, 1),
                             f"{len(wires)} wires so far, starting point {n_done} of {n_seeds}")
            y, x = ys[idx], xs[idx]
            k = int(self.R_arg[y, x])
            if self._occupied(k, x, y):
                continue
            phi = self.thetas[k]
            p0 = np.array([x, y], float)
            args = (low, step, max_turn, max_gap, search, max_steps, window, memory, recovery,
                    pass_length if pass_length is not None else 10 * w)
            fwd = self._follow(p0, phi, *args)
            bwd = self._follow(p0, phi + np.pi, *args)
            pts = np.array(bwd[::-1] + [p0] + fwd)
            if smooth and len(pts) > smooth:
                pts = smooth_polyline(pts, smooth)
            # Mark the wire as traced, in the orientation layers it occupies,
            # even if it is short (so its seeds are not traced again)
            self._mark(pts, occ_r)
            if len(pts) >= 2 and polyline_length(pts) >= 0.5 * min_length \
                    and self._isotropy(pts) <= max_isotropy \
                    and self._continuity(pts) >= min_continuity:
                wires.append(pts)
        self.raw_wires = wires
        self.wires = [w_ for w_ in wires]
        self.min_length = min_length
        return wires

    def _continuity(self, pts):
        """Median fraction of the contrast along a trace that survives the line opening."""
        if self._cont is None:
            return 1.0
        H, W = self.img.shape
        P = np.round(np.asarray(pts)).astype(int)
        xi, yi = np.clip(P[:, 0], 0, W - 1), np.clip(P[:, 1], 0, H - 1)
        bg = self._bg[yi, xi].astype(np.float32)
        c = self.img[yi, xi].astype(np.float32) - bg
        o = self._cont[yi, xi].astype(np.float32) - bg
        ok = c > 3
        if not ok.any():
            return 0.0
        return float(np.median(np.clip(o[ok] / c[ok], 0, 1)))

    def _isotropy(self, pts):
        """Median of R_perp / R_max along a trace: ~0 for a wire, ~1 for spots."""
        H, W = self.img.shape
        P = np.round(np.asarray(pts)).astype(int)
        xi, yi = np.clip(P[:, 0], 0, W - 1), np.clip(P[:, 1], 0, H - 1)
        rm = self.R_max[yi, xi].astype(np.float32)
        rp = self.R_perp[yi, xi].astype(np.float32)
        return float(np.median(rp / np.maximum(rm, 1e-3)))

    def _mark(self, pts, radius):
        """Set the occupancy bits along a traced wire (its layer and the two next)."""
        K = self.n_orientations
        H, W = self.img.shape
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            pts = np.vstack([pts, pts])
        lo = np.maximum(np.floor(pts.min(0)).astype(int) - radius - 1, 0)
        hi = np.minimum(np.ceil(pts.max(0)).astype(int) + radius + 2, [W, H])
        seg = np.diff(pts, axis=0)
        ang = np.arctan2(seg[:, 1], seg[:, 0])
        if not np.any(seg):
            bins = np.array([int(self.R_arg[int(pts[0, 1]), int(pts[0, 0])])])
        else:
            bins = np.round((ang % np.pi) / (np.pi / K)).astype(int) % K
        P = np.round(pts - lo).astype(np.int32)
        canvas = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
        region = self.occupancy[lo[1]:hi[1], lo[0]:hi[0]]
        dtype = self.occupancy.dtype.type
        for b in np.unique(bins):
            canvas[:] = 0
            sel = np.flatnonzero(bins == b)
            lines = [P[[i, i + 1]].reshape(-1, 1, 2) for i in sel]
            cv2.polylines(canvas, lines, False, 1, thickness=2 * radius + 1)
            on = canvas.astype(bool)
            for db in (-1, 0, 1):
                region[on] |= dtype(1 << ((b + db) % K))

    # ---------------------------------------------------------------- 3. joining
    def join(self, max_distance=None, max_angle_deg=20):
        """
        Join pieces of the same wire: two ends closer than ``max_distance``
        whose directions continue smoothly (each within ``max_angle_deg`` of
        the line joining them) are merged. Then drops wires shorter than
        ``min_length``.
        """
        w = self.wire_width
        max_distance = max_distance if max_distance is not None else 4 * w
        cos_max = np.cos(np.deg2rad(max_angle_deg))
        wires = [np.asarray(p, float) for p in self.wires]
        changed = True
        while changed:
            changed = False
            ends = []                                 # (wire, end, point, outward tangent)
            for i, p in enumerate(wires):
                for e in (0, 1):
                    ends.append((i, e, p[0] if e == 0 else p[-1], _end_tangent(p, e)))
            if len(ends) < 2:
                break
            pts = np.array([e[2] for e in ends])
            tree = cKDTree(pts)
            cands = []
            for a, b in tree.query_pairs(max_distance):
                ia, ea, pa, ta = ends[a]
                ib, eb, pb, tb = ends[b]
                if ia == ib:
                    continue
                # the two outward tangents must point at each other
                if np.dot(ta, -tb) < cos_max:
                    continue
                v = pb - pa
                dist = np.hypot(*v)
                if dist > 1.0:
                    v = v / dist
                    if np.dot(ta, v) < cos_max or np.dot(tb, -v) < cos_max:
                        continue
                ang = np.arccos(np.clip(np.dot(ta, -tb), -1, 1))
                cands.append((dist / max_distance + ang / np.deg2rad(max_angle_deg), a, b))
            cands.sort()
            used = set()
            merged = {}
            for _, a, b in cands:
                ia, ea = ends[a][:2]
                ib, eb = ends[b][:2]
                if a in used or b in used:
                    continue
                if ia in merged or ib in merged:     # one merge per wire per round
                    continue
                pa = wires[ia] if ea == 1 else wires[ia][::-1]   # ends at the joint
                pb = wires[ib] if eb == 0 else wires[ib][::-1]   # starts at the joint
                merged[ia] = merged[ib] = np.vstack([pa, pb])
                used.update((a, b))
            if merged:
                changed = True
                new = []
                done = set()
                for i, p in enumerate(wires):
                    if i in merged:
                        key = id(merged[i])
                        if key not in done:
                            new.append(merged[i])
                            done.add(key)
                    else:
                        new.append(p)
                wires = new
        self.wires = [p for p in wires if polyline_length(p) >= self.min_length]
        return self.wires

    def _low_background(self):
        """
        Level of the dark background around every pixel: the darkest value
        (of the slightly smoothed image) within ~5 wire widths, smoothed.
        Unlike a grayscale opening it does not rise inside bundles of wires,
        where the gaps between them are still dark. Kept at a reduced
        resolution (``self._lowbg``, ``self._lowbg_f``).
        """
        if getattr(self, "_lowbg", None) is not None:
            return self._lowbg, self._lowbg_f
        w = self.wire_width
        f = max(1, int(w // 1.5))
        img = self.img.astype(np.float32)
        H, W = img.shape
        small = cv2.resize(img, (max(1, W // f), max(1, H // f)), interpolation=cv2.INTER_AREA) \
            if f > 1 else img
        small = cv2.GaussianBlur(small, (0, 0), 1.0)
        r = max(2, int(np.ceil(5 * w / f)))
        low = cv2.erode(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        low = cv2.GaussianBlur(low, (0, 0), r / 2)
        self._lowbg, self._lowbg_f = low, f
        return low, f

    def _contrast_sampler(self, pts, margin):
        """
        Around the points ``pts``: a function ``f(q) -> (gray, background)``
        that samples (bilinear) the image and the dark background level
        (``_low_background``) at float positions. Unlike the ridge response,
        the image does not leak along a wire past its tip or across a gap.
        """
        H, W = self.img.shape
        m = int(np.ceil(margin)) + 3
        x0, y0 = max(int(pts[:, 0].min()) - m, 0), max(int(pts[:, 1].min()) - m, 0)
        x1, y1 = min(int(pts[:, 0].max()) + m + 1, W), min(int(pts[:, 1].max()) + m + 1, H)
        sub = self.img[y0:y1, x0:x1].astype(np.float32)
        low, f = self._low_background()
        h, w_ = sub.shape
        lh, lw = low.shape

        def bil(a, x, y):
            hh, ww = a.shape
            x = np.clip(x, 0, max(ww - 1.001, 0))
            y = np.clip(y, 0, max(hh - 1.001, 0))
            xi = np.minimum(x.astype(int), max(ww - 2, 0))
            yi = np.minimum(y.astype(int), max(hh - 2, 0))
            fx, fy = x - xi, y - yi
            xj, yj = np.minimum(xi + 1, ww - 1), np.minimum(yi + 1, hh - 1)
            return ((a[yi, xi] * (1 - fx) + a[yi, xj] * fx) * (1 - fy)
                    + (a[yj, xi] * (1 - fx) + a[yj, xj] * fx) * fy)

        def sample(q):
            q = np.atleast_2d(q)
            return (bil(sub, q[:, 0] - x0, q[:, 1] - y0),
                    bil(low, (q[:, 0] + 0.5) / f - 0.5, (q[:, 1] + 0.5) / f - 0.5))
        return sample

    def split_at_gaps(self, min_contrast=0.35, gap=None):
        """
        Cut the wires where the image shows no wire: a stretch of at least
        ``gap`` px (default max(2, width)) whose gray level is less than
        ``min_contrast`` x the wire's own contrast above the local background.

        The ridge filters are elongated and the tracer bridges short dim
        stretches (``max_gap``), and joining links aligned pieces, so a trace
        can run on from one wire to another across a dark gap. Each piece
        becomes a wire of its own; pieces shorter than ``min_length`` are
        dropped. Returns the number of cuts (also ``self.n_split``).
        """
        w = self.wire_width
        gap = gap if gap is not None else max(2.0, w)
        ds = 0.5
        n_gap = max(1, int(round(gap / ds)))
        min_len = getattr(self, "min_length", 0) or 0
        out, cuts = [], 0
        n_w = len(self.wires)
        for i, p in enumerate(self.wires):
            if i % 500 == 0:
                self._report("Closing gaps at wire ends", 0.0, f"checking wire {i} of {n_w}")
            p = np.asarray(p, float)
            s = resample_polyline(p, ds)
            if len(s) < 2 * n_gap + 4:
                out.append(p)
                continue
            g, bg = self._contrast_sampler(s, 2)(s)
            ref = float(np.median(g - bg))
            if not ref > 0:
                out.append(p)
                continue
            dark = (g - bg) < min_contrast * ref
            d = np.diff(np.r_[0, dark.astype(np.int8), 0])
            starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            keep = np.ones(len(s), bool)
            for a, e in zip(starts, ends):
                if e - a >= n_gap:
                    keep[a:e] = False
            if keep.all():
                out.append(p)
                continue
            d = np.diff(np.r_[0, keep.astype(np.int8), 0])
            pieces = [s[a:e] for a, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
            interior = sum(1 for a, e in zip(starts, ends) if e - a >= n_gap and a > 0 and e < len(s))
            cuts += interior
            out.extend(q for q in pieces if len(q) >= 2 and polyline_length(q) >= min_len)
        self.wires = out
        self.n_split = cuts
        self._wires_changed()
        return cuts

    def extend_ends(self, max_extension=None, min_contrast=0.5, max_trim=None, gap=1.5):
        """
        Fix the wire ends with the image itself, not the ridge response (the
        elongated filters keep responding a few px past a real tip and across
        short dark gaps, so the traces tend to overshoot).

        1. **Trim.** Near each end (within ``max_trim`` px, default
           max(6, 3 x width)), the wire is cut where the image goes dark: a run
           of at least ``gap`` px whose contrast is below ``min_contrast`` x
           the wire's own (median) contrast. Ends near an electrode, an
           excluded area or a mosaic margin are not trimmed.
        2. **Extend.** The tracer also stops a few px short of a junction,
           where the other wire distorts the ridge profile. From every end
           that does not touch another wire, walk straight ahead for at most
           ``max_extension`` px (default 3 x width) while the image stays
           bright (same contrast criterion, gaps shorter than ``gap`` px
           allowed). If another wire is reached, the end is extended up to it.

        Returns the number of ends extended (``self.n_trimmed`` keeps the
        number of ends trimmed).
        """
        w = self.wire_width
        max_ext = max_extension if max_extension is not None else 3 * w
        max_trim = max_trim if max_trim is not None else max(6.0, 3 * w)
        reach = getattr(self, "contact_distance", None) or 0.8 * w
        ds = 0.5
        n_gap = max(1, int(round(gap / ds)))
        self._extended_ends = [(False, False)] * len(self.wires)
        self.n_extended = self.n_trimmed = 0
        if not self.wires:
            return 0
        H, W = self.img.shape
        inside = getattr(self, "_inside", None)
        if inside is not None and inside.any():
            rr = int(round(5 + w + max_trim))
            protect = cv2.dilate(inside.astype(np.uint8),
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1,) * 2)) > 0
        else:
            protect = None

        def protected(q):
            if protect is None:
                return False
            return bool(protect[min(max(int(round(q[1])), 0), H - 1),
                                min(max(int(round(q[0])), 0), W - 1)])

        # 1. trim
        n_w = len(self.wires)
        wires, refs = [], []
        for i, p in enumerate(self.wires):
            if i % 500 == 0:
                self._report("Closing gaps at wire ends", 0.5 * i / n_w,
                             f"trimming, wire {i} of {n_w}")
            p = np.asarray(p, float)
            s = resample_polyline(p, ds)
            smp = self._contrast_sampler(s, max_ext)
            g, bg = smp(s)
            ref = float(np.median(g - bg))
            m = min(int(max_trim / ds), len(s) // 3)
            # dark background level near each end
            bends = (float(np.median(bg[:m + 1])), float(np.median(bg[len(s) - 1 - m:])))
            refs.append((ref, bends))
            if not ref > 0 or len(s) < 4 * n_gap:
                wires.append(p)
                continue
            a, b = 0, len(s) - 1
            for e in (0, 1):
                if protected(s[0] if e == 0 else s[-1]):
                    continue
                # samples from the inside of the wire towards the tip
                idx = np.arange(m, -1, -1) if e == 0 else np.arange(len(s) - 1 - m, len(s))
                d = (g[idx] - bends[e]) < min_contrast * ref
                run = 0
                for k, v in enumerate(d):
                    run = run + 1 if v else 0
                    if run >= n_gap:
                        cut = idx[k - run + 1]      # first dark sample of the run
                        if e == 0:
                            a = cut + 1
                        else:
                            b = cut - 1
                        break
            if a > 0 or b < len(s) - 1:
                self.n_trimmed += (a > 0) + (b < len(s) - 1)
                wires.append(s[a:b + 1])
            else:
                wires.append(p)
        self.wires = wires
        if len(wires) < 2:
            return 0

        # 2. extend
        samples = [resample_polyline(p, 1.0) for p in wires]
        P = np.vstack(samples)
        O = np.concatenate([np.full(len(s), i) for i, s in enumerate(samples)])
        tree = cKDTree(P)
        steps = np.arange(ds, max_ext + 1e-9, ds)
        new_wires, flags = [], []
        for i, p in enumerate(wires):
            if i % 500 == 0:
                self._report("Closing gaps at wire ends", 0.5 + 0.5 * i / n_w,
                             f"extending, wire {i} of {n_w}")
            ext = [False, False]
            smp = None
            ref, bends = refs[i]
            if ref > 0:
                for e in (0, 1):
                    thr = bends[e] + min_contrast * ref
                    end = p[0] if e == 0 else p[-1]
                    if any(O[j] != i for j in tree.query_ball_point(end, reach)):
                        continue        # already touching another wire
                    t = _end_tangent(p, e)
                    Q = end + steps[:, None] * t
                    ok = (Q[:, 0] >= 0) & (Q[:, 0] <= W - 1) & (Q[:, 1] >= 0) & (Q[:, 1] <= H - 1)
                    if smp is None:
                        smp = self._contrast_sampler(p, max_ext)
                    c = smp(Q)[0]
                    hit, run = None, 0
                    for q, v, o in zip(Q, c, ok):
                        if not o:
                            break
                        run = run + 1 if v < thr else 0
                        if run >= n_gap:
                            break
                        if any(O[j] != i for j in tree.query_ball_point(q, reach)):
                            hit = q
                            break
                    if hit is not None:
                        p = np.vstack([hit, p]) if e == 0 else np.vstack([p, hit])
                        ext[e] = True
                        self.n_extended += 1
            new_wires.append(p)
            flags.append(tuple(ext))
        self.wires = new_wires
        self._extended_ends = flags
        return self.n_extended

    # -------------------------------------------------------------- 4. contacts
    def find_contacts(self, contact_distance=None, end_tolerance=None):
        """
        ``self.contacts``: list of (i, j, x, y) for every pair of wires in
        contact, and ``self.electrode_contacts``: list of ("Vin"/"Vout", i, x, y).
        """
        self._wires_changed()
        return self._find_contacts(contact_distance, end_tolerance)

    def _find_contacts(self, contact_distance=None, end_tolerance=None):
        """
        ``self.contacts``: list of (i, j, x, y) for every pair of wires in
        contact, and ``self.electrode_contacts``: list of ("Vin"/"Vout", i, x, y).
        """
        w = self.wire_width
        cd = contact_distance if contact_distance is not None else 0.8 * w
        et = end_tolerance if end_tolerance is not None else 1.5 * w
        self.contact_distance, self.end_tolerance = cd, et
        samples, owner = [], []
        for i, p in enumerate(self.wires):
            s = resample_polyline(p, 1.0)
            samples.append(s)
            owner.append(np.full(len(s), i))
        if not samples:
            self.contacts, self.electrode_contacts = [], []
            return []
        P = np.vstack(samples)
        O = np.concatenate(owner)
        tree = cKDTree(P)
        best = {}                                     # (i, j) -> (dist, x, y)

        def _add(i, j, d, xy):
            key = (i, j) if i < j else (j, i)
            if key not in best or d < best[key][0]:
                best[key] = (d, xy[0], xy[1])

        pairs = tree.query_pairs(cd, output_type="ndarray")
        if len(pairs):
            pairs = pairs[O[pairs[:, 0]] != O[pairs[:, 1]]]
            d = np.hypot(*(P[pairs[:, 0]] - P[pairs[:, 1]]).T)
            for (a, b), dd in zip(pairs, d):
                _add(O[a], O[b], dd, (P[a] + P[b]) / 2)
        # a wire ending on (or just short of) another one
        for i, s in enumerate(samples):
            for e in (s[0], s[-1]):
                for b in tree.query_ball_point(e, et):
                    j = O[b]
                    if j != i:
                        _add(i, j, np.hypot(*(P[b] - e)), (P[b] + e) / 2)
        self.contacts = [(i, j, x, y) for (i, j), (_, x, y) in sorted(best.items())]

        self.electrode_contacts = []
        if self.has_electrodes:
            r = int(round(5 + w))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
            for name, m in (("Vin", self.electrode_mask1), ("Vout", self.electrode_mask2)):
                dil = cv2.dilate(m.astype(np.uint8), ker) > 0
                H, W = dil.shape
                for i, s in enumerate(samples):
                    xi = np.clip(np.round(s[:, 0]).astype(int), 0, W - 1)
                    yi = np.clip(np.round(s[:, 1]).astype(int), 0, H - 1)
                    hit = np.flatnonzero(dil[yi, xi])
                    if len(hit):
                        x, y = s[hit[len(hit) // 2]]
                        self.electrode_contacts.append((name, i, float(x), float(y)))
        return self.contacts

    # ------------------------------------------------------------ measurements
    def measure_wires(self, step=2.0, pixel_size=None):
        """
        Length, width and truncation of every wire, in ``self.measurements``
        (one dict per wire, same order as ``self.wires``).

        * ``length``: along the center line, in px (and ``length_um`` with
          ``pixel_size`` in micrometers per px).
        * ``width``: full width at half maximum of the intensity profile
          across the wire, in px (median along the wire, leaving out the
          places where another wire is closer than 2 widths: crossings would
          inflate it). NaN when it cannot be measured. In optical images of
          nanowires thinner than the resolution this is the width of the
          blurred image of the wire, not its diameter.
        * ``truncated``: an end lies at the image border, on an electrode, an
          excluded area or an empty mosaic margin, so the wire may continue
          beyond what is seen and ``length`` is only a lower bound. A wire
          split in pieces by the tracer is not flagged.
        """
        from scipy.ndimage import map_coordinates
        w = self.wire_width
        H, W = self.img.shape
        img = self.img.astype(np.float32)
        # areas where a wire cannot be followed
        blocked = np.zeros((H, W), bool)
        for m in (getattr(self, "electrode_mask1", None), getattr(self, "electrode_mask2", None),
                  self.exclude_mask, getattr(self, "margin_mask", None)):
            if m is not None:
                blocked |= m
        r = int(round(2 * w)) + 1
        if blocked.any():
            blocked = cv2.dilate(blocked.astype(np.uint8),
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
        border = 2 * w
        # samples along every wire
        samples = [resample_polyline(p, step) for p in self.wires]
        owner = np.concatenate([np.full(len(s_), i) for i, s_ in enumerate(samples)]) \
            if samples else np.zeros(0, int)
        P = np.vstack(samples) if samples else np.zeros((0, 2))
        T = np.vstack([np.gradient(s_, axis=0) if len(s_) > 1 else np.array([[1.0, 0.0]])
                       for s_ in samples]) if samples else np.zeros((0, 2))
        T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
        widths = np.full(len(P), np.nan)
        if len(P):
            # skip samples with another wire nearby
            dense = [resample_polyline(p, 1.0) for p in self.wires]
            D = np.vstack(dense)
            DO = np.concatenate([np.full(len(d), i) for i, d in enumerate(dense)])
            tree = cKDTree(D)
            _, idx = tree.query(P, k=12, distance_upper_bound=2 * w)
            valid = idx < len(D)
            others = valid & (DO[np.minimum(idx, len(D) - 1)] != owner[:, None])
            alone = ~others.any(axis=1)
            # profiles across the wire
            half = int(np.ceil(3 * w + 3))
            t = np.arange(-half, half + 1e-9, 0.25)
            N = np.c_[-T[:, 1], T[:, 0]]
            X = P[:, 0:1] + t[None] * N[:, 0:1]
            Y = P[:, 1:2] + t[None] * N[:, 1:2]
            prof = map_coordinates(img, [Y.ravel(), X.ravel()], order=1,
                                   mode="nearest").reshape(X.shape)
            c0 = len(t) // 2
            k = max(1, int(round(0.5 / 0.25)))          # peak within +-0.5 px
            win = prof[:, c0 - k:c0 + k + 1]
            ip = c0 - k + win.argmax(1)
            peak = prof[np.arange(len(prof)), ip]
            nb = max(4, len(t) // 8)
            bl = np.sort(prof[:, :nb * 2], axis=1)[:, :nb].mean(1)
            br = np.sort(prof[:, -nb * 2:], axis=1)[:, :nb].mean(1)
            hl, hr = (peak + bl) / 2, (peak + br) / 2
            cols = np.arange(len(t))[None]
            below_l = (prof < hl[:, None]) & (cols < ip[:, None])
            below_r = (prof < hr[:, None]) & (cols > ip[:, None])
            okl, okr = below_l.any(1), below_r.any(1)
            jl = np.where(okl, len(t) - 1 - below_l[:, ::-1].argmax(1), 0)
            jr = np.where(okr, below_r.argmax(1), 0)
            rows = np.arange(len(prof))
            pl0, pl1 = prof[rows, jl], prof[rows, np.minimum(jl + 1, len(t) - 1)]
            pr0, pr1 = prof[rows, np.maximum(jr - 1, 0)], prof[rows, jr]
            xl = t[jl] + (hl - pl0) / np.where(pl1 != pl0, pl1 - pl0, 1) * 0.25
            xr = t[np.maximum(jr - 1, 0)] + (pr0 - hr) / np.where(pr0 != pr1, pr0 - pr1, 1) * 0.25
            contrast = peak - np.maximum(bl, br)
            good = alone & okl & okr & (contrast >= max(5.0, self.low_threshold if hasattr(self, "low_threshold") else 5.0))
            widths[good] = (xr - xl)[good]
        self.measurements = []
        for i, p in enumerate(self.wires):
            p = np.asarray(p)
            wv = widths[owner == i]
            wv = wv[np.isfinite(wv)]
            trunc = False
            for e in (p[0], p[-1]):
                x, y = e
                if x < border or y < border or x > W - 1 - border or y > H - 1 - border:
                    trunc = True
                elif blocked[min(int(round(y)), H - 1), min(int(round(x)), W - 1)]:
                    trunc = True
            length = self._visible_length(p, i, img)
            m = dict(length=length, width=float(np.median(wv)) if len(wv) >= 3 else float("nan"),
                     truncated=trunc)
            if pixel_size:
                m["length_um"] = length * pixel_size
                m["width_um"] = m["width"] * pixel_size
            self.measurements.append(m)
        self.pixel_size = pixel_size
        return self.measurements

    def set_pixel_size(self, pixel_size):
        """
        Set (or clear, with None) the scale after measuring: updates
        ``length_um`` / ``width_um`` of every wire and the graph attributes.
        """
        self.pixel_size = pixel_size or None
        for m in getattr(self, "measurements", None) or []:
            m.pop("length_um", None)
            m.pop("width_um", None)
            if self.pixel_size:
                m["length_um"] = m["length"] * self.pixel_size
                m["width_um"] = m["width"] * self.pixel_size
        L = getattr(self, "L", None)
        if L is not None:
            for n, d in L.nodes(data=True):
                if isinstance(n, str):
                    continue
                d.pop("length_um", None)
                d.pop("width_um", None)
                m = self.measurements[n]
                for k in ("length_um", "width_um"):
                    if k in m and not np.isnan(m[k]):
                        d[k] = float(m[k])

    def _visible_length(self, p, i, img):
        """
        Length of the wire as seen in the image: without the bridges added by
        ``extend_ends`` and with each end cut where the brightness along the
        wire falls below half of its contrast (the ridge filter keeps
        responding a few px past the real tip).
        """
        ext = self._extended_ends[i] if getattr(self, "_extended_ends", None) and \
            i < len(self._extended_ends) else (False, False)
        p = np.asarray(p, float)
        if ext[0] and len(p) > 2:
            p = p[1:]
        if ext[1] and len(p) > 2:
            p = p[:-1]
        s = resample_polyline(p, 0.5)
        if len(s) < 5:
            return polyline_length(p)
        H, W = img.shape
        xi = np.clip(np.round(s[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(s[:, 1]).astype(int), 0, H - 1)
        r = int(round(1.5 * self.wire_width)) + 1
        y0, y1 = max(yi.min() - r, 0), min(yi.max() + r + 1, H)
        x0, x1 = max(xi.min() - r, 0), min(xi.max() + r + 1, W)
        sub = self.img[y0:y1, x0:x1]
        bg = cv2.morphologyEx(sub, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        c = img[yi, xi] - bg[yi - y0, xi - x0].astype(np.float32)
        half = 0.5 * np.median(c)
        if not half > 0:
            return polyline_length(p)
        on = np.flatnonzero(c >= half)
        if len(on) < 2:
            return polyline_length(p)
        a, b = on[0], on[-1]
        # unless the end was bridged on purpose, cut where the tip fades
        a = 0 if ext[0] else a
        b = len(s) - 1 if ext[1] else b
        return polyline_length(s[a:b + 1])

    def measurement_table(self, path, only_graph=True):
        """Write the measurements as CSV (one row per wire; node id = ``wire``)."""
        import csv
        ids = [n for n in self.L.nodes if not isinstance(n, str)] if only_graph \
            else range(len(self.wires))
        keys = ["length", "width", "truncated"]
        if getattr(self, "pixel_size", None):
            keys += ["length_um", "width_um"]
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["wire", "degree"] + keys)
            for i in ids:
                m = self.measurements[i]
                deg = self.L.degree(i) if i in self.L else 0
                wr.writerow([i, deg] + [m[k] for k in keys])

    def measurement_summary(self, only_graph=True):
        """Medians and quartiles of length and width (complete wires for length)."""
        ids = [n for n in self.L.nodes if not isinstance(n, str)] if only_graph \
            else range(len(self.wires))
        M = [self.measurements[i] for i in ids]
        L_all = np.array([m["length"] for m in M])
        L_ok = np.array([m["length"] for m in M if not m["truncated"]])
        Wd = np.array([m["width"] for m in M])
        Wd = Wd[np.isfinite(Wd)]
        q = lambda a: tuple(np.percentile(a, [25, 50, 75])) if len(a) else (np.nan,) * 3  # noqa: E731
        return dict(n=len(M), n_truncated=int(len(L_all) - len(L_ok)),
                    length_q=q(L_ok), length_all_q=q(L_all), width_q=q(Wd), n_width=len(Wd))

    # ---------------------------------------------------------------- 5. graphs
    def _wires_changed(self):
        """Forget what was cached about the wires (after they are traced or edited)."""
        self.__dict__.pop("_arc_cache", None)
        self.__dict__.pop("_path_cache", None)

    def build_graphs(self, drop_dangling=True):
        """
        Build ``self.L`` (nodes = wires, edges = contacts) and the spatial graph
        ``self.G`` (nodes = contacts, edges = pieces of wire between them).

        With ``drop_dangling`` wires with fewer than two contacts are removed
        repeatedly (the equivalent of pruning degree-1 nodes in
        ``SkeletonAnalysis.clean_network``: a wire touching a single other wire
        cannot carry current). Electrode nodes are always kept.
        """
        self._drop_dangling = drop_dangling
        L = nx.Graph()
        removed = getattr(self, "removed_wires", set())
        for i, p in enumerate(self.wires):
            if i in removed:
                continue
            p = np.asarray(p)
            c = p.mean(axis=0)
            paths = self.__dict__.setdefault("_path_cache", {})
            if i not in paths:
                paths[i] = ";".join(f"{x:.1f},{y:.1f}" for x, y in resample_polyline(p, 3.0))
            meas = self.measurements[i] if getattr(self, "measurements", None) else {}
            extra = {k: (float(v) if not isinstance(v, bool) else v) for k, v in meas.items()
                     if k != "length" and not (isinstance(v, float) and np.isnan(v))}
            L.add_node(i, x=float(c[0]), y=float(c[1]),
                       length=float(meas.get("length", polyline_length(p))), **extra,
                       x0=float(p[0, 0]), y0=float(p[0, 1]),
                       x1=float(p[-1, 0]), y1=float(p[-1, 1]),
                       path=paths[i])
        for i, j, x, y in self.contacts:
            if i in removed or j in removed:
                continue
            L.add_edge(i, j, x=float(x), y=float(y))
        self.electrode_pos = {}
        counters = {"Vin": 0, "Vout": 0}
        for name, i, x, y in self.electrode_contacts:
            if i in removed:
                continue
            node = f"{name}_{counters[name]}"
            counters[name] += 1
            L.add_node(node, x=float(x), y=float(y))
            L.add_edge(node, i, x=float(x), y=float(y))
            self.electrode_pos[node] = (x, y)
        L.remove_edges_from([(u, v) for u, v in L.edges if isinstance(u, str) and isinstance(v, str)])
        self.L_full = L.copy()
        if drop_dangling:
            while True:
                weak = [n for n, d in L.degree() if d < 2 and not isinstance(n, str)]
                if not weak:
                    break
                L.remove_nodes_from(weak)
            L.remove_nodes_from([n for n, d in L.degree() if d == 0])
        self.L = L
        self.G = self._spatial_graph(L)
        return L

    def remove_wires(self, ids, drop_dangling=None):
        """
        Delete wires by hand (false detections): they and their contacts leave
        the graph, which is rebuilt (dangling wires dropped again if the last
        build dropped them). Undo with ``restore_wires``.
        """
        self.removed_wires |= {int(i) for i in ids}
        self._rebuild(drop_dangling)

    def restore_wires(self, ids=None):
        """Undo ``remove_wires`` for ``ids`` (all of them if None)."""
        if ids is None:
            self.removed_wires = set()
        else:
            self.removed_wires -= {int(i) for i in ids}
        self._rebuild(None)

    def _rebuild(self, drop_dangling):
        if drop_dangling is None:
            drop_dangling = getattr(self, "_drop_dangling", True)
        self.build_graphs(drop_dangling=drop_dangling)

    def wire_at(self, x, y, radius=None):
        """Index of the wire whose center line passes closest to (x, y), or None."""
        radius = radius if radius is not None else max(3.0, 1.5 * self.wire_width)
        best, bd = None, radius
        for i, p in enumerate(self.wires):
            if i in self.removed_wires:
                continue
            p = np.asarray(p)
            if (p[:, 0].min() - radius > x or p[:, 0].max() + radius < x or
                    p[:, 1].min() - radius > y or p[:, 1].max() + radius < y):
                continue
            d = np.hypot(*(resample_polyline(p, 1.0) - (x, y)).T).min()
            if d < bd:
                best, bd = i, d
        return best

    def _spatial_graph(self, L):
        """Nodes = contacts (and electrode nodes); edges = wire pieces between them."""
        G = nx.Graph()
        coords = []
        self.electrode_pos = getattr(self, "electrode_pos", {})
        on_wire = {i: [] for i in L.nodes if not isinstance(i, str)}
        for u, v, d in L.edges(data=True):
            if isinstance(u, str) or isinstance(v, str):
                node, wire = (u, v) if isinstance(u, str) else (v, u)
                G.add_node(node)
                on_wire[wire].append((d["x"], d["y"], node))
                continue
            nid = len(coords)
            coords.append((d["y"], d["x"]))
            G.add_node(nid)
            on_wire[u].append((d["x"], d["y"], nid))
            on_wire[v].append((d["x"], d["y"], nid))
        # position of each contact along its wire; cached, as it does not change
        # when wires are deleted by hand and the graph is rebuilt
        cache = self.__dict__.setdefault("_arc_cache", {})
        for wire, items in on_wire.items():
            if len(items) < 2:
                continue
            missing = [t for t in items if (wire, t[0], t[1]) not in cache]
            if missing:
                p = resample_polyline(self.wires[wire], 1.0)
                _, idx = cKDTree(p).query([(t[0], t[1]) for t in missing])
                for t, k in zip(missing, np.atleast_1d(idx)):
                    cache[(wire, t[0], t[1])] = int(k)
            order = sorted(items, key=lambda t: cache[(wire, t[0], t[1])])
            for (_, _, a), (_, _, b) in zip(order[:-1], order[1:]):
                if a != b:
                    G.add_edge(a, b, wire=int(wire))
        self.node_coords = np.array(coords, float).reshape(-1, 2)
        return G

    # ---------------------------------------------------------------- export
    def save_graph(self, path, save_line_graph=True):
        """
        Export to .graphml: the wire graph ``L`` (default; nodes = wires) or
        the spatial graph ``G`` (nodes = contacts, with ``x``/``y``).
        """
        if save_line_graph:
            H = self.L.copy()
        else:
            H = self.G.copy()
            for n in H.nodes:
                if n in self.electrode_pos:
                    x, y = self.electrode_pos[n]
                else:
                    y, x = self.node_coords[n]
                H.nodes[n]["x"], H.nodes[n]["y"] = float(x), float(y)
        H.graph["image_height"], H.graph["image_width"] = map(int, self.img.shape[:2])
        H.graph["method"] = "wire_tracing"
        nx.write_graphml(H, path)

    def wire_label_image(self, thickness=None, wires=None):
        """Image with each wire drawn with its index + 1 (0 = background)."""
        thickness = thickness or max(1, int(round(self.wire_width)))
        lab = np.zeros(self.img.shape[:2], np.int32)
        ids = range(len(self.wires)) if wires is None else wires
        for i in ids:
            p = np.round(self.wires[i]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(lab, [p], False, int(i) + 1, thickness=thickness)
        return lab

    def free_memory(self):
        """Drop the cached orientation scores and the occupancy map."""
        if hasattr(self, "_cache"):
            self._cache.clear()
        for a in ("occupancy", "_cont", "_bg", "_lowbg"):
            if hasattr(self, a):
                delattr(self, a)

    # ---------------------------------------------------------------- pipeline
    def complete_analysis(self, electrodes_path=None, clean_network=True, save_graph_path=None,
                          n_orientations=16, seed_threshold=None, low_threshold=None,
                          max_gap=2, join_distance=None, min_length=None,
                          contact_distance=None, end_tolerance=None, extend_ends=True,
                          split_gaps=True,
                          continuity_length=None, exclude_mask=None, pixel_size=None,
                          keep_scores=False, verbose=False, progress=None):
        """
        Run the whole pipeline. Returns ``self.L``. ``progress``, if given, is
        called as ``progress(stage, fraction, detail)`` while it runs
        (fraction of the whole analysis, 0 to 1), a few times per second.
        """
        import time
        t0 = time.time()

        def log(msg):
            if verbose:
                print(f"[{time.time() - t0:6.1f} s] {msg}")

        if progress is not None:
            self.progress = progress

        if electrodes_path:
            self.load_electrodes(*electrodes_path)
            log("Electrode masks loaded.")
        if exclude_mask is not None:
            self.load_exclusion(exclude_mask)
            log("Exclusion mask loaded.")
        self.orientation_scores(n_orientations=n_orientations,
                                continuity_length=continuity_length)
        log(f"Orientation scores ({n_orientations} orientations), noise = {self.noise:.1f}.")
        self.trace(seed_threshold=seed_threshold, low_threshold=low_threshold,
                   max_gap=max_gap, min_length=min_length)
        log(f"Traced {len(self.raw_wires)} pieces "
            f"(seed >= {self.seed_threshold:.1f}, continue >= {self.low_threshold:.1f}).")
        self._report("Joining pieces of wires", 0, f"{len(self.raw_wires)} pieces", force=True)
        self.join(max_distance=join_distance)
        log(f"Joined into {len(self.wires)} wires.")
        if split_gaps:
            self._report("Closing gaps at wire ends", 0, f"{len(self.wires)} wires", force=True)
            n = self.split_at_gaps()
            log(f"Cut {n} wires where the trace crossed a dark gap ({len(self.wires)} wires).")
        if extend_ends:
            self._report("Closing gaps at wire ends", 0, f"{len(self.wires)} wires", force=True)
            n = self.extend_ends()
            log(f"Trimmed {self.n_trimmed} wire ends that overshot their tip; "
                f"extended {n} up to the wire they run into.")
        self._report("Finding contacts", 0, f"{len(self.wires)} wires", force=True)
        self.find_contacts(contact_distance=contact_distance, end_tolerance=end_tolerance)
        log(f"{len(self.contacts)} wire-wire contacts, "
            f"{len(self.electrode_contacts)} electrode contacts.")
        self._report("Measuring wires", 0, f"{len(self.wires)} wires", force=True)
        self.measure_wires(pixel_size=pixel_size)
        self._report("Building the graph", 0, f"{len(self.contacts)} contacts", force=True)
        self.build_graphs(drop_dangling=clean_network)
        log(f"Wire graph: {self.L.number_of_nodes()} nodes, {self.L.number_of_edges()} edges.")
        if not keep_scores:
            self.free_memory()
        if save_graph_path:
            self.save_graph(save_graph_path)
            log(f"Saved to {save_graph_path}.")
        self._report("Building the graph", 1.0, "done", force=True)
        return self.L

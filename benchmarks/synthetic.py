"""
Synthetic nanowire micrographs with known ground truth, to measure how
faithfully a method recovers the wires (one node per wire) and their contacts.

    python benchmarks/synthetic.py --out bench_data --n-wires 150 350 --seeds 0 1 2

Each image ``<name>.png`` comes with ``<name>_wires.npz`` holding the wire
polylines (``wire_<i>`` arrays of (x, y) points) and the rendering width.
"""

import argparse
import os

import cv2
import numpy as np


def random_wire(rng, shape, length_range=(60, 320), curvature=0.007, step=1.0):
    """A smooth random curve whose curvature (1/radius, per px) drifts slowly."""
    h, w = shape
    L = rng.uniform(*length_range)
    n = max(int(L / step), 2)
    x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
    # Preferred direction (real samples are often aligned), with spread
    th0 = rng.normal(0.0, 0.6) if rng.random() < 0.6 else rng.uniform(0, np.pi)
    kappa = rng.normal(0, curvature) + rng.normal(0, curvature * 0.08, n).cumsum()
    kappa = np.clip(kappa, -0.025, 0.025)            # radius >= 40 px
    th = th0 + np.cumsum(kappa) * step
    pts = np.empty((n, 2))
    pts[0] = (x0, y0)
    pts[1:] = pts[0] + np.cumsum(np.c_[np.cos(th[1:]), np.sin(th[1:])] * step, axis=0)
    # centre the wire on (x0, y0)
    pts -= (pts[-1] - pts[0]) / 2
    return pts


def render(wires, shape, width=3.2, rng=None, noise=0.04, background=0.12,
           brightness=(0.35, 1.0), blur=0.7):
    """Dark-field-like rendering: bright wires, darker uneven background."""
    rng = rng or np.random.default_rng(0)
    h, w = shape
    ss = 4                                            # supersampling
    acc = np.zeros((h, w), np.float32)
    for pts in wires:
        b = rng.uniform(*brightness)
        layer = np.zeros((h * ss, w * ss), np.uint8)
        p = np.round(pts * ss).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(layer, [p], False, 255, thickness=max(int(round(width * ss)), 1),
                      lineType=cv2.LINE_AA)
        layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        # Wires on top look brighter where they overlap (additive with saturation)
        acc = 1 - (1 - acc) * (1 - layer * b)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    bg = background * (0.6 + 0.8 * (0.5 * np.sin(3 * xx + 1) + 0.5) * (0.7 + 0.3 * yy))
    img = bg + acc * (1 - bg)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), blur)
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def make(n_wires, seed, shape=(1024, 1024), width=3.2):
    rng = np.random.default_rng(seed)
    wires = [random_wire(rng, shape) for _ in range(n_wires)]
    h, w = shape
    clipped = []
    for p in wires:                                   # keep the part inside the image
        inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        if inside.sum() < 25:
            continue
        idx = np.flatnonzero(inside)
        # longest run of consecutive inside points
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = max(runs, key=len)
        if len(run) >= 25:
            clipped.append(p[run])
    img = render(clipped, shape, width=width, rng=rng)
    return img, clipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_data")
    ap.add_argument("--n-wires", type=int, nargs="+", default=[150, 350])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--size", type=int, default=1024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for n in a.n_wires:
        for s in a.seeds:
            img, wires = make(n, s, shape=(a.size, a.size))
            name = f"syn_n{n}_s{s}"
            cv2.imwrite(os.path.join(a.out, name + ".png"), img)
            np.savez_compressed(os.path.join(a.out, name + "_wires.npz"), width=3.2,
                                **{f"wire_{i}": p for i, p in enumerate(wires)})
            print(name, len(wires), "wires")


if __name__ == "__main__":
    main()

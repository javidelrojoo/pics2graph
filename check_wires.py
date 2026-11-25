"""
check_wires
===========

Visual check of the wire reconstruction: shows random wires one at a time
(the wire in red, the wires it touches in light blue, the contacts in
yellow) and asks whether each one is right, to estimate the fraction of
correctly reconstructed wires.

    python check_wires.py                                  # example sample M4
    python check_wires.py --background path/M4.tif         # original micrograph as background
    python check_wires.py --image path/M5_binary.tif \\
        --electrodes path/M5_electrode_mask1.tif path/M5_electrode_mask2.tif

Close each window to answer in the console: b = right, p = split (the wire
continues in another color), f = merged (the red wire jumps onto another
wire), c = wrong contact, q = quit. The wrong ones are printed with their
position so they can be looked at again.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from SkeletonAnalysis import SkeletonAnalysis


def parse_args():
    p = argparse.ArgumentParser(description="Check reconstructed wires one at a time.")
    p.add_argument("--image", default="example/M4/M4_binary.tif", help="Binary image.")
    p.add_argument("--electrodes", nargs=2, metavar=("VIN", "VOUT"),
                   default=["example/M4/M4_electrode_mask1.tif",
                            "example/M4/M4_electrode_mask2.tif"])
    p.add_argument("--no-electrodes", action="store_true")
    p.add_argument("--background", default=None,
                   help="Original micrograph to use as background (recommended).")
    p.add_argument("-n", type=int, default=30, help="Number of wires to check (default 30).")
    p.add_argument("--min-length", type=float, default=50,
                   help="Only wires at least this long, in px (default 50).")
    p.add_argument("--max-angle", type=float, default=45.0)
    p.add_argument("--opening-radius", type=int, default=1)
    p.add_argument("--closing-radius", type=int, default=0)
    p.add_argument("--min-object-size", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def show_wire(SA, W, k, bg, title, margin=40):
    pts = np.concatenate(SA.wires[k])
    cy, cx = pts.mean(axis=0)
    r = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + margin
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(bg, cmap="gray", alpha=0.6)
    for n in W.neighbors(k):
        if n in SA.wires:
            ax.add_collection(LineCollection([q[:, ::-1] for q in SA.wires[n]],
                                             colors="deepskyblue", linewidths=1.5))
        d = W[k][n]
        ax.scatter(d["x"], d["y"], s=40, c="yellow", edgecolors="k", zorder=3)
    ax.add_collection(LineCollection([q[:, ::-1] for q in SA.wires[k]],
                                     colors="red", linewidths=2.5))
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy + r, cy - r)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.show()
    return cx, cy


def main():
    a = parse_args()
    electrodes = None if a.no_electrodes else tuple(a.electrodes)
    SA = SkeletonAnalysis(a.image)
    if SA.img is None:
        raise FileNotFoundError(f"Could not read the image: {a.image}")
    SA.complete_analysis(electrodes_path=electrodes, opening_radius=a.opening_radius,
                         closing_radius=a.closing_radius, min_object_size=a.min_object_size,
                         wire_options={"max_angle": a.max_angle})
    W = SA.W
    bg = plt.imread(a.background) if a.background else SA.img_original

    cands = [k for k in W if k in SA.wires and W.nodes[k]["length"] >= a.min_length]
    if not cands:
        print("No wires to check.")
        return
    picks = np.random.default_rng(a.seed).choice(cands, min(a.n, len(cands)), replace=False)
    print(f"{W.number_of_nodes()} wires in the graph; checking {len(picks)}.")

    answers = {"b": 0, "p": 0, "f": 0, "c": 0}
    wrong = []
    for i, k in enumerate(picks, 1):
        title = (f"[{i}/{len(picks)}] wire {k}: {W.nodes[k]['length']:.0f} px, "
                 f"{W.degree(k)} contacts - close to answer")
        cx, cy = show_wire(SA, W, k, bg, title)
        resp = ""
        while resp not in ("b", "p", "f", "c", "q"):
            resp = input(f"Wire {k}: b = right, p = split, f = merged, "
                         f"c = wrong contact, q = quit: ").strip().lower()
        if resp == "q":
            break
        answers[resp] += 1
        if resp != "b":
            wrong.append((k, resp, cx, cy))

    total = sum(answers.values())
    if not total:
        return
    print(f"\nRight: {answers['b']}/{total} ({100 * answers['b'] / total:.0f} %)"
          f"  |  split: {answers['p']}, merged: {answers['f']}, wrong contact: {answers['c']}")
    for k, resp, cx, cy in wrong:
        print(f"  wire {k} ({resp}) at x={cx:.0f}, y={cy:.0f}")


if __name__ == "__main__":
    main()

"""
Fidelity of the extracted graph against the ground truth of the synthetic
images (``synthetic.py``), for the new wire tracing and for the original
binarize + skeletonize pipeline.

    python benchmarks/evaluate.py bench_data/syn_n150_s0.png [more images...]

Metrics (per image):
* nodes            : nodes of the wire graph (wires; for the skeleton
                     pipeline, wire segments = nodes of its line graph).
* whole wires      : fraction of true wires recovered by a single node
                     covering >= 80 % of their length.
* nodes per wire   : mean number of nodes matched to each recovered true wire
                     (1.0 = one node per wire; higher = wires split in pieces).
* spurious nodes   : fraction of nodes that do not lie on a single true wire.
* contacts P / R   : precision / recall of the wire-wire contacts, after
                     mapping each node to the true wire it lies on.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from WireTracing import WireAnalysis, resample_polyline  # noqa: E402


def load_truth(png):
    d = np.load(png[:-4] + "_wires.npz")
    wires = [d[k] for k in sorted((k for k in d.files if k.startswith("wire_")),
                                  key=lambda s: int(s.split("_")[1]))]
    return wires, float(d["width"])


def true_contacts(wires, shape, width):
    """Contacts between the true wires, with the same rule as WireAnalysis."""
    wa = WireAnalysis(img=np.zeros(shape, np.uint8), wire_width=width)
    wa.wires = wires
    wa.find_contacts()
    return {(min(i, j), max(i, j)) for i, j, _, _ in wa.contacts}


def match(pred, truth, tol=2.0):
    """
    Assign each predicted polyline to the true wire it lies on.
    Returns (assignment list, per-true-wire list of (pred, coverage)).
    """
    ts = [resample_polyline(p, 1.0) for p in truth]
    T = np.vstack(ts)
    TO = np.concatenate([np.full(len(s), i) for i, s in enumerate(ts)])
    ttree = cKDTree(T)
    assign = []
    covers = {i: [] for i in range(len(truth))}
    for k, p in enumerate(pred):
        s = resample_polyline(p, 1.0)
        if len(s) == 0:
            assign.append(None)
            continue
        # nearest true points; on crossings several wires are close: use all within tol
        lists = ttree.query_ball_point(s, tol)
        votes = {}
        for lst in lists:
            for o in set(TO[lst]):
                votes[o] = votes.get(o, 0) + 1
        if not votes:
            assign.append(None)
            continue
        g, nv = max(votes.items(), key=lambda t: t[1])
        purity = nv / len(s)
        if purity < 0.7:
            assign.append(None)
            continue
        assign.append(int(g))
        # coverage of the true wire by this node
        ptree = cKDTree(s)
        dd, _ = ptree.query(ts[g])
        covers[g].append((k, float((dd <= tol).mean())))
    return assign, covers


def score(pred, pred_edges, truth, gt_edges):
    assign, covers = match(pred, truth)
    n_true = len(truth)
    whole = sum(1 for i in range(n_true) if any(c >= 0.8 for _, c in covers[i])) / n_true
    useful = {i: [k for k, c in covers[i] if c >= 0.1] for i in range(n_true)}
    recovered = [i for i in range(n_true) if useful[i]]
    per_wire = np.mean([len(useful[i]) for i in recovered]) if recovered else float("nan")
    covered = sum(1 for i in range(n_true)
                  if (lambda cs: cs and 1 - np.prod([1 - c for _, c in cs]) >= 0.5)(covers[i])) / n_true
    spurious = sum(a is None for a in assign) / max(len(pred), 1)
    mapped = set()
    for a, b in pred_edges:
        ga, gb = assign[a], assign[b]
        if ga is None or gb is None or ga == gb:
            continue
        mapped.add((min(ga, gb), max(ga, gb)))
    tp = len(mapped & gt_edges)
    prec = tp / len(mapped) if mapped else float("nan")
    rec = tp / len(gt_edges) if gt_edges else float("nan")
    return dict(nodes=len(pred), whole=whole, per_wire=per_wire, covered=covered,
                spurious=spurious, prec=prec, rec=rec)


# ----------------------------------------------------------------------------
# Original pipeline, keeping the pixel path of every edge through the cleaning
# ----------------------------------------------------------------------------
def skeleton_pipeline(img, block_size=31, C=-8, thresh=0, closing=1, opening=0, min_obj=10):
    import preprocessing as pre
    from SkeletonAnalysis import SkeletonAnalysis
    mask = pre.adaptive_binarize(img, block_size=block_size, C=C, intensity_thresh=thresh)
    SA = SkeletonAnalysis(img=mask)
    SA.img_cleaning(opening_radius=opening, min_object_size=min_obj, closing_radius=closing)
    SA.skeletonize()
    SA.get_network()
    G = SA.G
    path = {}
    for u, v, d in G.edges(data=True):
        p = np.asarray(d["path"])[:, ::-1]          # (row, col) -> (x, y)
        path[frozenset((u, v))] = p
    # Same steps as SkeletonAnalysis.clean_network, carrying the paths along
    changed = True
    while changed:
        changed = False
        iso = [n for n, dg in G.degree() if dg == 0]
        if iso:
            G.remove_nodes_from(iso)
            changed = True
        d1 = [n for n, dg in G.degree() if dg == 1]
        if d1:
            G.remove_nodes_from(d1)
            changed = True
        d2 = [n for n, dg in G.degree() if dg == 2]
        for n in d2:
            nb = list(G.neighbors(n))
            if len(nb) == 2:
                u, v = nb
                if not G.has_edge(u, v):
                    a = path.get(frozenset((u, n)), np.empty((0, 2)))
                    b = path.get(frozenset((n, v)), np.empty((0, 2)))
                    path[frozenset((u, v))] = np.vstack([a, b])
                G.add_edge(u, v)
            G.remove_node(n)
            changed = True
    edges = list(G.edges())
    idx = {frozenset(e): k for k, e in enumerate(edges)}
    polys = [path.get(frozenset(e), np.empty((0, 2))) for e in edges]
    L = nx.line_graph(G)
    pe = [(idx[frozenset(a)], idx[frozenset(b)]) for a, b in L.edges()]
    return polys, pe


def run(png, **kw):
    img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    truth, width = load_truth(png)
    gt_edges = true_contacts(truth, img.shape, width)
    out = {"image": os.path.basename(png), "true wires": len(truth), "true contacts": len(gt_edges)}

    t = time.time()
    WA = WireAnalysis(img=img, wire_width=width)
    WA.complete_analysis(clean_network=False, **kw)
    wires = WA.wires
    edges = [(i, j) for i, j, _, _ in WA.contacts]
    out["new"] = score(wires, edges, truth, gt_edges)
    out["new"]["time"] = time.time() - t

    t = time.time()
    polys, pe = skeleton_pipeline(img)
    out["old"] = score(polys, pe, truth, gt_edges)
    out["old"]["time"] = time.time() - t
    return out, WA


def fmt(r):
    return (f"{r['nodes']:6d} nodes | whole wires {r['whole']:5.1%} | nodes/wire {r['per_wire']:4.2f} | "
            f"covered {r['covered']:5.1%} | spurious {r['spurious']:5.1%} | "
            f"contacts P {r['prec']:5.1%} R {r['rec']:5.1%} | {r['time']:5.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    for png in a.images:
        out, _ = run(png)
        print(f"{out['image']}: {out['true wires']} wires, {out['true contacts']} contacts")
        print("  skeleton :", fmt(out["old"]))
        print("  tracing  :", fmt(out["new"]))


if __name__ == "__main__":
    main()

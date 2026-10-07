"""
main
====

Entry point of the image -> graph pipeline.

Takes a photomicrograph (already binarized, or raw with ``--binarize``) and,
optionally, the electrode masks, extracts the network with
``SkeletonAnalysis`` and saves the graph as .graphml.

Examples
--------
    # Example sample included in the repo (binary image + electrodes)
    python main.py

    # Your own binary image, without electrodes
    python main.py --image path/M5_binary.tif --output M5.graphml

    # Raw image: binarize first, then extract the graph
    python main.py --image path/M5.tif --binarize --block-size 151 --C 15 --threshold 55

    # With electrodes, also saving the figure of the graph over the image
    python main.py --image path/M5_binary.tif \
        --electrodes path/M5_electrode_mask1.tif path/M5_electrode_mask2.tif \
        --save-plot M5_graph.png
"""

import argparse
import os

import cv2

import preprocessing as pre
from SkeletonAnalysis import SkeletonAnalysis

# Defaults: the example sample included in ./example/M4
SAMPLE = "M4"
DIR_PATH = os.path.join("example", SAMPLE)
IMG_PATH = os.path.join(DIR_PATH, f"{SAMPLE}_binary.tif")
ELECTRODES_PATH = (
    os.path.join(DIR_PATH, f"{SAMPLE}_electrode_mask1.tif"),
    os.path.join(DIR_PATH, f"{SAMPLE}_electrode_mask2.tif"),
)
SAVE_GRAPH_PATH = os.path.join(DIR_PATH, f"{SAMPLE}_effective.graphml")


def parse_args():
    p = argparse.ArgumentParser(description="Convert a photomicrograph into a graph (.graphml).")
    p.add_argument("--image", default=IMG_PATH,
                   help="Input image (binary, or raw when using --binarize).")
    p.add_argument("--electrodes", nargs=2, metavar=("VIN", "VOUT"), default=None,
                   help="Masks of the input and output electrodes.")
    p.add_argument("--no-electrodes", action="store_true",
                   help="Do not use electrodes (even if there are defaults).")
    p.add_argument("--output", default=None, help="Path of the output .graphml.")
    p.add_argument("--graph", choices=("line", "spatial"), default="line",
                   help="Which graph to save: the line graph (nodes = wires, default) "
                        "or the spatial graph (nodes = junctions).")
    p.add_argument("--no-clean", action="store_true",
                   help="Do not prune degree-1 nodes nor collapse degree-2 nodes.")

    b = p.add_argument_group("binarization (only with --binarize)")
    b.add_argument("--binarize", action="store_true",
                   help="The image is raw: binarize it before extracting the graph.")
    b.add_argument("--block-size", type=int, default=151)
    b.add_argument("--C", type=int, default=15)
    b.add_argument("--threshold", type=int, default=55, help="intensity_thresh")

    v = p.add_argument_group("visualization")
    v.add_argument("--plot", action="store_true", help="Show the graph over the image.")
    v.add_argument("--save-plot", default=None, help="Save the figure to this path.")
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    using_default = args.image == IMG_PATH

    img_path = args.image
    if args.binarize:
        img = pre.load_gray(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read the image: {img_path}")
        mask = pre.adaptive_binarize(img, block_size=args.block_size, C=args.C,
                                     intensity_thresh=args.threshold)
        base, _ = os.path.splitext(img_path)
        img_path = f"{base}_binary.tif"
        cv2.imwrite(img_path, mask)
        if verbose:
            print(f"Binarized image saved to {img_path}.")

    electrodes = args.electrodes
    if electrodes is None and using_default:
        electrodes = ELECTRODES_PATH
    if args.no_electrodes:
        electrodes = None

    output = args.output
    if output is None:
        output = SAVE_GRAPH_PATH if using_default else os.path.splitext(img_path)[0] + ".graphml"

    SA = SkeletonAnalysis(img_path)
    if SA.img is None:
        raise FileNotFoundError(f"Could not read the image: {img_path}")

    SA.complete_analysis(
        clean_network=not args.no_clean,
        electrodes_path=tuple(electrodes) if electrodes else None,
        save_graph_path=None,
        plot=False,
        verbose=verbose,
    )
    SA.save_graph(output, save_line_graph=(args.graph == "line"))
    if verbose:
        G, L = SA.G, SA.L
        print(f"Spatial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        print(f"Line graph: {L.number_of_nodes()} nodes, {L.number_of_edges()} edges.")
        print(f"Graph ({args.graph}) saved to {output}.")

    if args.plot or args.save_plot:
        import matplotlib.pyplot as plt
        from visualization import plot_network_on_image
        plot_network_on_image(SA, save_path=args.save_plot, dpi=400)
        if args.plot:
            plt.show()

    return SA


if __name__ == "__main__":
    main()

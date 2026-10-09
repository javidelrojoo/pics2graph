"""
main
====

Entry point of the image -> graph pipeline.

Takes a photomicrograph (already binarized, or raw with ``--binarize``), or a
folder of tiles to stitch first with Fiji (``--stitch``), and optionally the
electrode masks; extracts the network with ``SkeletonAnalysis`` and saves the
graph as .graphml.

Examples
--------
    # Example sample included in the repo (binary image + electrodes)
    python main.py

    # Your own binary image, without electrodes
    python main.py --image path/M5_binary.tif --output M5.graphml

    # Raw image: binarize first, then extract the graph
    python main.py --image path/M5.tif --binarize --block-size 151 --C 15 --threshold 55

    # Folder of tiles: stitch with Fiji, binarize, then extract the graph
    python main.py --stitch path/M5_tiles --fiji C:/fiji-stable-win64-jdk/Fiji.app

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
from stitching import stitch

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
    p.add_argument("--opening-radius", type=int, default=1,
                   help="Radius of the morphological opening before skeletonizing "
                        "(default 1; use 0 when the wires are 1-2 px wide, e.g. SEM "
                        "stitched with --stitch-downsample 2 or more).")
    p.add_argument("--min-object-size", type=int, default=0,
                   help="Remove objects of up to this many pixels before skeletonizing.")
    p.add_argument("--closing-radius", type=int, default=0,
                   help="Radius of a morphological closing before skeletonizing, to keep "
                        "thin noisy wires continuous (SEM: 1 at downsample 4, 2 at downsample 2).")

    s = p.add_argument_group("stitching (Fiji, see stitching.py for all options)")
    s.add_argument("--stitch", metavar="TILES_DIR", default=None,
                   help="Folder of tiles: stitch them with Fiji first (implies --binarize). "
                        "The mosaic is saved as <TILES_DIR>/Fused.tif unless --image is given.")
    s.add_argument("--fiji", default=None, help="Fiji.app folder or launcher (or set FIJI_PATH).")
    s.add_argument("--stitch-mode", choices=("unknown", "positions"), default="unknown",
                   help="'unknown': the plugin finds the arrangement (default); "
                        "'positions': use <TILES_DIR>/TileConfiguration.txt.")
    s.add_argument("--fiji-memory", default=None, help="Java heap for Fiji, e.g. 8g.")
    s.add_argument("--stitch-downsample", type=int, default=1,
                   help="Reduce each tile by this factor before stitching (large sets, e.g. SEM).")
    s.add_argument("--stitch-gray", action="store_true",
                   help="Convert the tiles to 8-bit gray before stitching.")

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
    if args.stitch:
        using_default = False
        mosaic = None if args.image == IMG_PATH else args.image
        img_path = stitch(args.stitch, output_path=mosaic, mode=args.stitch_mode,
                          fiji_path=args.fiji, memory=args.fiji_memory,
                          downsample=args.stitch_downsample, tiles_to_gray=args.stitch_gray,
                          verbose=verbose)
        args.binarize = True

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
        opening_radius=args.opening_radius,
        min_object_size=args.min_object_size,
        closing_radius=args.closing_radius,
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

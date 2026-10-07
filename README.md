# pics2graph

Converts photomicrographs of self-assemblies (nanowire networks) into **connection graphs** (`.graphml`).

Extracted from the thesis repository *"Obtaining connection graphs of neuromorphic self-assemblies from their photomicrographs"* (Javier Tau Anzoátegui, LINE, ECyT-UNSAM). It contains only what is needed to go from **image → graph**; network metrics, robustness experiments and the memristor simulation remain in the thesis repository.

## Pipeline

```
raw image ──(preprocessing)──► binary image ──(SkeletonAnalysis)──► skeleton ──► spatial graph G ──► line graph L ──► .graphml
                                                   ▲
                                electrode masks (optional, Vin / Vout)
```

| File | Role |
| --- | --- |
| `preprocessing.py` | Adaptive binarization of the photomicrograph. |
| `SkeletonAnalysis.py` | Morphological cleaning, skeletonization, graph extraction and pruning, line graph and export. |
| `visualization.py` | Plots the graph (or the line graph) overlaid on the image. |
| `main.py` | Command-line entry point. |
| `example/M4/` | Example sample: binary image and the masks of the two electrodes. |

## Installation

```bash
pip install -r requirements.txt
```

## Quick start (command line)

```bash
# Run on the example sample (M4 with electrodes); saves example/M4/M4_effective.graphml
python main.py

# Your own binary image, without electrodes
python main.py --image path/M5_binary.tif --output M5.graphml

# Raw image: it is binarized first (saved as path/M5_binary.tif), then the graph is extracted
python main.py --image path/M5.tif --binarize --block-size 151 --C 15 --threshold 55

# With electrodes, also saving the figure of the graph over the image
python main.py --image path/M5_binary.tif \
    --electrodes path/M5_electrode_mask1.tif path/M5_electrode_mask2.tif \
    --save-plot M5_graph.png
```

Main options (`python main.py -h` lists them all):

| Option | Description |
| --- | --- |
| `--image` | Input image (binary, or raw with `--binarize`). |
| `--electrodes VIN VOUT` | Masks of the input and output electrodes. |
| `--no-electrodes` | Ignore electrodes. |
| `--output` | Path of the `.graphml` (defaults to next to the image). |
| `--graph {line,spatial}` | Graph to save: line graph (nodes = wires, default) or spatial graph (nodes = junctions). |
| `--no-clean` | Do not prune degree-1 nodes nor collapse degree-2 nodes. |
| `--binarize`, `--block-size`, `--C`, `--threshold` | Prior adaptive binarization and its parameters. |
| `--plot`, `--save-plot` | Show / save the graph over the image. |

## Usage from Python

```python
import preprocessing as pre
from SkeletonAnalysis import SkeletonAnalysis
from visualization import plot_network_on_image

# (optional) binarize a raw image
img = pre.load_gray("M5.tif")
mask = pre.adaptive_binarize(img, block_size=151, C=15, intensity_thresh=55)
overlay = pre.overlay_mask(img, mask)   # to check the binarization

# extract the graph
SA = SkeletonAnalysis("example/M4/M4_binary.tif")
SA.complete_analysis(
    electrodes_path=("example/M4/M4_electrode_mask1.tif", "example/M4/M4_electrode_mask2.tif"),
    save_graph_path="M4.graphml",
    verbose=True,
)
SA.G   # spatial graph (networkx): nodes = junctions, edges = wire segments
SA.L   # line graph: nodes = wires, edges = contacts between wires

plot_network_on_image(SA, background="skeleton")
```

Step by step, instead of `complete_analysis`:

```python
SA.img_cleaning()
SA.load_electrodes(path_vin, path_vout)   # optional, before skeletonize
SA.skeletonize()
SA.get_network()
SA.clean_network()
SA.convert_to_line_graph()
SA.save_graph("output.graphml", save_line_graph=True)
```

## Reference

### `preprocessing.py`

* **`load_gray(path)`**: loads an image in grayscale.
* **`adaptive_binarize(img, block_size=151, C=15, intensity_thresh=55)`**: Gaussian adaptive threshold (with replicated padding) that also forces pixels below `intensity_thresh` to background. Returns the mask (`0`/`255`).
* **`overlay_mask(base_gray, binary_mask, color_rgb=(220, 20, 60))`**: overlays the mask in color on the image for inspection.

### `SkeletonAnalysis`

* **`__init__(img_path)`**: loads the binarized image.
* **`load_electrodes(path1, path2)`**: loads the Vin and Vout masks (fills holes and removes small objects).
* **`img_cleaning()`**: removes small holes and applies an opening with a disk of radius 1.
* **`skeletonize()`**: reduces structures to 1-pixel width; if there are electrodes, removes them from the skeleton and keeps dilated masks to connect them.
* **`get_network()`**: converts the skeleton into a graph (`skan` → networkx) and adds `Vin_i` / `Vout_i` nodes where the skeleton touches the electrodes.
* **`clean_network()`**: iteratively removes isolated nodes, prunes degree-1 nodes, collapses degree-2 nodes and removes edges between electrodes.
* **`convert_to_line_graph()`**: builds `SA.L`, the line graph of `SA.G`.
* **`save_graph(path, save_line_graph=True)`**: exports to `.graphml`.
* **`plot_graph(save_path=None)`**: quick plot of the graph over the image.
* **`complete_analysis(clean_network=True, electrodes_path=None, save_graph_path=None, plot=False, save_plot_path=None, verbose=False)`**: runs the whole pipeline.

### `visualization.py`

* **`plot_network_on_image(SA, xlim=None, ylim=None, background="skeleton", ...)`**: draws `SA.G` over the image (Vin in red, Vout in blue). `background` can be `"skeleton"`, `"img"`, `"raw"`, `"original"`, a path or an array. `xlim`/`ylim` zoom in and scale nodes and edges accordingly.
* **`plot_line_graph_on_image(SA, ...)`**: same, for the line graph (each node at the centroid of its wire).
* **`get_node_positions(SA)`** / **`get_line_graph_positions(SA)`**: `(x, y)` node positions for plotting with networkx.

## Notes

* The input image to `SkeletonAnalysis` must be binarized (wires in white), preferably as `.tif`.
* Compatible with scikit-image both before and after 0.26 (change from `min_size`/`area_threshold` to `max_size`).
* With the example sample, `python main.py` reproduces the thesis' `M4_effective.graphml` (6703 nodes and 13242 edges in the line graph).

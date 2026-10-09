# pics2graph

Converts photomicrographs of self-assemblies (nanowire networks) into **connection graphs** (`.graphml`).

Extracted from the thesis repository *"Obtaining connection graphs of neuromorphic self-assemblies from their photomicrographs"* (Javier Tau Anzoátegui, LINE, ECyT-UNSAM). It contains only what is needed to go from **image → graph**; network metrics, robustness experiments and the memristor simulation remain in the thesis repository.

## Pipeline

```
tiles ──(stitching, Fiji)──► raw image ──(preprocessing)──► binary image ──(SkeletonAnalysis)──► skeleton ──► spatial graph G ──► line graph L ──► .graphml
                                                                         ▲
                                                      electrode masks (optional, Vin / Vout)
```

| File | Role |
| --- | --- |
| `stitching.py` | Stitches overlapping tiles into one mosaic by running Fiji's *Grid/Collection stitching* plugin headless. |
| `preprocessing.py` | Adaptive binarization of the photomicrograph. |
| `SkeletonAnalysis.py` | Morphological cleaning, skeletonization, graph extraction and pruning, line graph and export. |
| `visualization.py` | Plots the graph (or the line graph) overlaid on the image. |
| `main.py` | Command-line entry point. |
| `gui.py` | Graphical interface (napari) covering the whole pipeline. |
| `example/M4/` | Example sample: binary image and the masks of the two electrodes. |

## Installation

```bash
pip install -r requirements.txt
```

Stitching additionally needs a local [Fiji](https://fiji.sc) installation (the Stitching plugin is included by default). It is found automatically in common locations (e.g. `C:/fiji-*/Fiji.app`, `~/Fiji.app`, `/Applications/Fiji.app`); otherwise pass `--fiji path/to/Fiji.app` or set the `FIJI_PATH` environment variable once:

```bash
setx FIJI_PATH "C:\fiji-stable-win64-jdk\Fiji.app"     # Windows (open a new terminal afterwards)
export FIJI_PATH=/path/to/Fiji.app                     # Linux / macOS
```

## Graphical interface

```bash
python gui.py                     # empty viewer
python gui.py path/to/M5.tif      # open a micrograph directly
```

On Windows you can also **double-click `pics2graph.bat`** (no need to open a terminal first), or drag an image onto it to open that image. It opens a terminal window that shows the loading progress and any error messages; if the program ends with an error, that window stays open until you press a key. To launch it from the Desktop, right-click `pics2graph.bat` › *Send to* › *Desktop (create shortcut)*.

It opens a [napari](https://napari.org) viewer with a *pics2graph* panel on the right, one tab per stage:

1. **Image**: open a micrograph or an already binarized image, or stitch a folder of tiles with Fiji (all the plugin parameters of `stitching.py` are available; Fiji's progress is printed in the terminal).
2. **Binarize**: adaptive binarization with live preview while you move *block size*, *C* and the intensity threshold. The result is an editable `binary` layer: select it and use napari's brush (label 1) to add wire pixels or the eraser to remove them. *Save binary image...* exports it.
3. **Electrodes**: paint the input electrode (Vin, red) and the output electrode (Vout, blue) with the brush, polygon or fill tools, or load existing masks. *Save masks...* writes `<name>_electrode_mask1.tif` / `_electrode_mask2.tif`, compatible with the command line.
4. **Graph**: extract the graph (optional cleaning and electrodes), view the spatial graph G or the line graph L over the image with zoom, and export the `.graphml`, a figure or a screenshot.

Startup can take a while, especially the first time after installing or rebooting (Python compiles and caches napari and Qt, and on Windows the antivirus scans those files); later starts take a few seconds. A small loading window and messages in the terminal show each step meanwhile.

Every layer can be hidden, made more transparent or edited with napari's own tools (left panel). Long tasks (stitching, graph extraction) run in the background, so the window stays responsive.

## Quick start (command line)

```bash
# Run on the example sample (M4 with electrodes); saves example/M4/M4_effective.graphml
python main.py

# Your own binary image, without electrodes
python main.py --image path/M5_binary.tif --output M5.graphml

# Raw image: it is binarized first (saved as path/M5_binary.tif), then the graph is extracted
python main.py --image path/M5.tif --binarize --block-size 151 --C 15 --threshold 55

# Folder of tiles: stitch with Fiji -> binarize -> graph
# (saves path/M5_tiles/Fused.tif, Fused_binary.tif and Fused_binary.graphml)
python main.py --stitch path/M5_tiles

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
| `--stitch TILES_DIR` | Stitch the tiles in that folder with Fiji first (implies `--binarize`). |
| `--fiji`, `--stitch-mode {unknown,positions}`, `--fiji-memory` | Fiji location, stitching mode and Java memory. |
| `--stitch-downsample N`, `--stitch-gray` | Reduce the tiles N times / convert them to 8-bit gray before stitching (large sets). |
| `--binarize`, `--block-size`, `--C`, `--threshold` | Prior adaptive binarization and its parameters. |
| `--plot`, `--save-plot` | Show / save the graph over the image. |

## Stitching (`stitching.py`)

The stitching is done by Fiji itself: Python writes an ImageJ macro with the chosen parameters, runs Fiji headless (no window), waits for it and collects the fused image. The result is the same as running *Plugins › Stitching › Grid/Collection stitching* by hand. The tiles are copied to a temporary folder first, so your folder is not modified (apart from the output) and non-image files are ignored. Files named `Fused*`/`stitched*` and images with a different size than the tiles (e.g. a mosaic from a previous run) are skipped automatically.

```bash
# Default: "Unknown position" (the plugin works out the arrangement by itself)
python stitching.py path/M5_tiles -o path/M5.tif

# Regular grid acquisition, e.g. snake by rows, 2 columns x 3 rows, ~30 % overlap
python stitching.py path/M5_tiles --mode grid --grid-type snake-by-rows --order right-down \
    --grid-size 2 3 --overlap 30 --file-names "Image_{iiiii}.tif" --first-index 30938

# Approximate positions from a TileConfiguration file
python stitching.py path/M5_tiles --mode positions --layout-file path/M5_tiles/TileConfiguration.txt

# Large sets (e.g. 180 SEM tiles of 8192 x 6144 px): reduce 4x and convert to 8-bit gray
python stitching.py path/SEM_tiles --downsample 4 --tiles-to-gray

python stitching.py -h      # all options
```

**Large tile sets.** ImageJ cannot hold an image larger than 2^31 − 1 pixels (~2150 Mpx), and the fusion needs the whole mosaic in memory, so very large sets fail with *Out of memory* no matter how much RAM is available. `stitch` estimates the mosaic size before running Fiji and stops with a suggested factor if it cannot fit. Use:

* `downsample=n`: each tile is reduced n times (each n × n block of pixels is averaged) before stitching, so the mosaic has n² times fewer pixels. Check that the wires stay visible: in SEM images with 4–6 px wide wires, `downsample=4` still resolves them (1–2 px).
* `tiles_to_gray=True`: 8-bit grayscale tiles (3× less memory than color, 2× less than 16-bit). Palette images whose used colors are all gray (typical of SEM, e.g. Zeiss) are always converted to gray.
* `computation="auto"` (default): switches to Fiji's low-memory mode when the tiles add up to more than 2 GB.

As a reference, for 180 tiles of 8192 × 6144 px: full resolution ~7250 Mpx (does not fit), `downsample=2` ~1800 Mpx (fits, but the later binarization and graph extraction are heavy), `downsample=4` ~450 Mpx (comfortable on 16 GB of RAM). In the GUI, the *Check size* button gives this estimate for the current settings.

When the tiles need no changes they are hard-linked instead of copied (no extra disk space); working folders left by interrupted runs are removed automatically after a few hours.

From Python:

```python
from stitching import stitch

mosaic = stitch("path/M5_tiles", output_path="path/M5.tif",
                fusion_method="Linear Blending", regression_threshold=0.30,
                max_avg_displacement_threshold=2.50, absolute_displacement_threshold=3.50)
```

* **`stitch(tiles_dir, output_path=None, mode="unknown", ...)`**: runs the plugin and saves the mosaic (default `<tiles_dir>/Fused.tif`) plus the registered positions (`<name>_TileConfiguration.registered.txt`). Returns the mosaic path.
  * `mode`: `"unknown"` (*Unknown position*), `"grid"` (*Grid: row-by-row / column-by-column / snake by rows / snake by columns*, with `grid_type`, `order`, `grid_size`, `tile_overlap`, `file_names`, `first_index`) or `"positions"` (*Positions from file*, with `layout_file`).
  * Plugin parameters: `fusion_method`, `regression_threshold`, `max_avg_displacement_threshold`, `absolute_displacement_threshold`, `compute_overlap`, `subpixel_accuracy`, `computation` (`"time"` / `"memory"`). Anything else can be passed verbatim with `extra_options` (copy it from Fiji's macro recorder).
  * Large sets: `downsample`, `tiles_to_gray`, `check_size` (see above).
  * Other: `pattern` / `tiles` to choose the tiles, `grayscale`, `fiji_path`, `memory` (Java heap, e.g. `"8g"`), `timeout`, `keep_temp`, `verbose`.
* **`estimate_mosaic(tiles, downsample=1, tiles_to_gray=False)`**: estimated mosaic size and the smallest/suggested `downsample`, without running Fiji.
* **`find_fiji(fiji_path=None)`**: locates the Fiji launcher.
* **`build_options(...)` / `build_macro(...)`**: the option string and macro passed to Fiji (useful to inspect what will be run).

> Since the right parameters depend on each set of images, start with the defaults and, if a mosaic looks wrong, adjust the thresholds or switch to `grid`/`positions` mode. With `verbose=True` Fiji's log (correlations of each pair of tiles, discarded links, final displacements) is printed to the console.

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

* The stitching was checked against a mosaic made by hand in Fiji (sample M4, 6 tiles): `stitch(...)` with default parameters gives the same 2767 × 2424 image (mean difference 0.3 gray levels out of 255).
* `SkeletonAnalysis` accepts the binary image and the electrode masks either as file paths or as arrays (`SkeletonAnalysis(img=mask)`, `load_electrodes(mask1, mask2)`).
* The project uses `opencv-python-headless` (OpenCV without its own windows), which avoids Qt conflicts with napari. If you already had `opencv-python` installed, uninstall it first: `pip uninstall opencv-python`.
* The input image to `SkeletonAnalysis` must be binarized (wires in white), preferably as `.tif`.
* Compatible with scikit-image both before and after 0.26 (change from `min_size`/`area_threshold` to `max_size`).
* With the example sample, `python main.py` reproduces the thesis' `M4_effective.graphml` (6703 nodes and 13242 edges in the line graph).

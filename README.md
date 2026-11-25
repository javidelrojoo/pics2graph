# pics2graph

Converts photomicrographs of self-assemblies (nanowire networks) into **connection graphs** (`.graphml`).

Extracted from the thesis repository *"Obtaining connection graphs of neuromorphic self-assemblies from their photomicrographs"* (Javier Tau Anzoátegui, LINE, ECyT-UNSAM). It contains only what is needed to go from **image → graph**; network metrics, robustness experiments and the memristor simulation remain in the thesis repository.

## Pipeline

Two methods turn the image into a graph:

```
                                     ┌──(WireTracing, default)──────────────────────────────────────► wires ──► L: node = whole wire ──► .graphml
tiles ──(stitching, Fiji)──► raw image
                                     └──(preprocessing)──► binary image ──(SkeletonAnalysis)──► skeleton ──► G ──► L: node = wire segment ──► .graphml
                                                                    ▲
                                                electrode masks (optional, Vin / Vout), for both methods
```

* **Wire tracing** (`WireTracing.py`, default) works directly on the grayscale micrograph: no binarization and no skeleton. It follows each wire through the crossings, so **each node of the graph is a whole wire** and each edge is a contact between two wires. See *Wire tracing* below.
* **Skeleton** (`SkeletonAnalysis.py`, `--method skeleton`) is the method of the thesis: binarize, skeletonize, and take the line graph of the skeleton. A node is a **segment** of wire between two junctions, so a wire crossed by others is split into several nodes, and the junction where several segments meet becomes a clique.

| File | Role |
| --- | --- |
| `electrode_registration.py` | Finds the electrodes of a later photo (other magnification/rotation) on the existing image and turns them into electrode masks. |
| `stitching.py` | Stitches overlapping tiles into one mosaic by running Fiji's *Grid/Collection stitching* plugin headless. |
| `preprocessing.py` | Adaptive binarization of the photomicrograph. |
| `WireTracing.py` | Graph of whole wires traced on the grayscale micrograph (default method). |
| `SkeletonAnalysis.py` | Morphological cleaning, skeletonization, graph extraction and pruning, line graph and export (skeleton method). |
| `visualization.py` | Plots the graph (or the line graph) overlaid on the image. |
| `main.py` | Command-line entry point. |
| `gui.py` | Graphical interface (napari) covering the whole pipeline. |
| `benchmarks/` | Synthetic micrographs with known wires, to measure how faithful each method is. |
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
2. **Binarize** (only needed for the skeleton method): adaptive binarization with live preview while you move *block size*, *C* and the intensity threshold. The result is an editable `binary` layer: select it and use napari's brush (label 1) to add wire pixels or the eraser to remove them. *Save binary image...* exports it.
3. **Electrodes**: paint the input electrode (Vin, red) and the output electrode (Vout, blue) with the brush, polygon or fill tools, load existing masks, or get them **from a later photo** (see *Electrodes from a later photo*). *Save masks...* writes `<name>_electrode_mask1.tif` / `_electrode_mask2.tif`, compatible with the command line.
4. **Graph**: choose the method (*Trace wires on the micrograph*, the default, or *Skeleton of the binary image*), extract the graph (optional cleaning and electrodes), view it over the image with zoom, and export the `.graphml`, a figure or a screenshot. With wire tracing the *Wires* view draws each wire in its own color (hover over one to see its node id + 1), which is the quickest way to check that every node is one whole wire; its parameters (wire width, start/end contrast, gap, join and contact distances) default to *auto*.

Startup can take a while, especially the first time after installing or rebooting (Python compiles and caches napari and Qt, and on Windows the antivirus scans those files); later starts take a few seconds. A small loading window and messages in the terminal show each step meanwhile.

Every layer can be hidden, made more transparent or edited with napari's own tools (left panel). Long tasks (stitching, graph extraction) run in the background, so the window stays responsive. Wire tracing shows a progress bar with the current stage (filtering, tracing, joining, contacts, measurements), how many wires it has found so far and an estimate of the time left; the command line prints the same every 5 %.

## Quick start (command line)

```bash
# Raw micrograph -> graph of whole wires (saves path/M5_wires.graphml)
python main.py --image path/M5.tif

# With electrodes, also saving the figure of the wires over the image
python main.py --image path/M5.tif \
    --electrodes path/M5_electrode_mask1.tif path/M5_electrode_mask2.tif \
    --save-plot M5_wires.png

# Folder of tiles: stitch with Fiji -> graph of wires
# (saves path/M5_tiles/Fused.tif and Fused_wires.graphml)
python main.py --stitch path/M5_tiles

# Skeleton method on the example sample (M4 with electrodes);
# reproduces the thesis' example/M4/M4_effective.graphml
python main.py --method skeleton

# Skeleton method on a raw image: binarized first (saved as path/M5_binary.tif)
python main.py --method skeleton --image path/M5.tif --binarize \
    --block-size 151 --C 15 --threshold 55
```

Main options (`python main.py -h` lists them all):

| Option | Description |
| --- | --- |
| `--method {wires,skeleton}` | Wire tracing on the micrograph (default; nodes = whole wires) or the skeleton pipeline (nodes = wire segments). |
| `--image` | Input image: the micrograph for `wires`; binary, or raw with `--binarize`, for `skeleton`. |
| `--electrodes VIN VOUT` | Masks of the input and output electrodes. |
| `--no-electrodes` | Ignore electrodes. |
| `--output` | Path of the `.graphml` (defaults to next to the image). |
| `--graph {line,spatial}` | Graph to save: line graph (nodes = wires, default) or spatial graph (nodes = junctions / contacts). |
| `--no-clean` | Keep dangling parts: wires with fewer than two contacts (`wires`), degree-1 and degree-2 nodes (`skeleton`). |
| `--wire-width`, `--seed-contrast`, `--min-contrast`, `--max-gap`, `--join-distance`, `--contact-distance`, `--orientations`, `--invert`, `--exclude`, `--pixel-size`, `--measurements` | Wire tracing parameters (see *Wire tracing*); all have automatic defaults. |
| `--opening-radius`, `--closing-radius`, `--min-object-size` | Cleaning before skeletonizing (see *Binarization and cleaning parameters*). |
| `--stitch TILES_DIR` | Stitch the tiles in that folder with Fiji first (with `--method skeleton` it implies `--binarize`). |
| `--fiji`, `--stitch-mode {unknown,positions}`, `--fiji-memory` | Fiji location, stitching mode and Java memory. |
| `--stitch-downsample N`, `--stitch-gray` | Reduce the tiles N times / convert them to 8-bit gray before stitching (large sets). |
| `--binarize`, `--block-size`, `--C`, `--threshold` | Prior adaptive binarization and its parameters (skeleton method). |
| `--plot`, `--save-plot` | Show / save the graph over the image. |

## Wire tracing (graph of whole wires)

`WireTracing.WireAnalysis` builds the graph straight from the grayscale micrograph. Binarizing throws away the information that tells two crossing wires apart (after thresholding, a crossing is just a blob), and the skeleton then cuts every wire at every crossing, so the old graph has one node per *segment*. Instead:

1. **Orientation scores.** The image is filtered with a bank of elongated ridge filters (second derivative of an anisotropic Gaussian) at 16 orientations. For every pixel this gives how strongly a wire runs through it *in each direction*, in gray levels of contrast above the local background (a smooth background gives no response, so uneven illumination does not matter). Where two wires cross, each one keeps responding in its own orientation.
2. **Tracing.** Starting from the strongest ridge points, each wire is followed in both directions in steps of 1.5 px, re-centered on its ridge, looking only at orientations close to its current heading and with a limited curvature. At a crossing the other wire is in a different orientation, so the tracer goes straight through. Short dim stretches are bridged (`max_gap`), sharper bends are tried when the ridge is lost straight ahead, and a wire already traced in the same direction is not traced again.
   Bright structure that is not a wire is ignored. Round spots (dust, particles, blurred dots) respond as much across as along any direction, so they cannot start a wire. Rows of spots can look like a line, but they are not *continuous*: a grayscale opening of the image with straight segments of 5 × width, in every orientation, keeps the contrast of wires and erases spots and rows of spots with darker gaps; seeds and traces must keep at least 60 % of their contrast after it. Last, the empty margins of a stitched mosaic (regions that are exactly 0 and touch the border), whose straight edges would otherwise be traced as wires.
3. **Joining.** Pieces of the same wire that could not be followed in one go (shallow crossings, bright clumps, dim spots) are joined when their ends are close and their directions continue smoothly.
4. **Ends.** The tracer tends to stop a few px short of a junction, where the other wire distorts the ridge. Each end that does not touch anything is extended straight ahead, up to 3 × width, as long as the image stays bright; if it reaches another wire, the gap is closed.
5. **Contacts.** Two wires touch when their center lines come within `contact_distance` (crossings, side contacts) or an end of one is within `end_tolerance` of the other (a wire ending on another). Wires touching an electrode mask get `Vin_i` / `Vout_i` neighbors, named as before, so the downstream code that looks for `"Vin"` / `"Vout"` in the node names keeps working.
6. **Cleaning.** Wires with fewer than two contacts are removed repeatedly (they cannot carry current), the equivalent of pruning degree-1 nodes in the skeleton method. They are still drawn, in gray, in the *Wires* view and in `plot_wires_on_image`, so a wire left out of the graph can be told apart from one that was not detected.

Each node of `L` stores the wire's centroid (`x`, `y`), `length` in px, endpoints (`x0`, `y0`, `x1`, `y1`) and the whole center line as `path` (`"x,y;x,y;..."`, every 3 px); each edge stores the contact position (`x`, `y`). `G` (`--graph spatial`) has the contacts as nodes and the pieces of wire between consecutive contacts as edges.

**Parameters.** All have automatic defaults; these are the ones worth adjusting while looking at the *Wires* view:

| Parameter (Python / CLI) | Default | When to change it |
| --- | --- | --- |
| `wire_width` / `--wire-width` | measured on the image | If the estimate printed is clearly off. Too large merges close wires; too small splits wires. |
| `seed_threshold` / `--seed-contrast` | max(10, 6 × noise) | Raise it if dust or noise is traced as short wires. |
| `low_threshold` / `--min-contrast` | max(5, 3 × noise) | Lower it if dim wires are cut short; raise it if traces run off the end of wires. |
| `max_gap` / `--max-gap` | 2 steps (3 px) | Raise it for wires with dim spots. |
| `join_distance` / `--join-distance` | 4 × width | Raise it if wires stay split at crossings; lower it if different wires get chained. |
| `contact_distance` / `--contact-distance` | 0.8 × width | Distance between center lines counted as contact. |
| `invert` / `--invert` | off | Wires darker than the background (bright field). |
| `exclude_mask` / `--exclude` | none | Mask (white = ignore) of areas that are still traced but are not part of the network, such as dust or substrate around it. In the GUI: *Paint areas to ignore* in tab 4. |

Wires must be brighter than the background (optical dark field, SEM), otherwise use `invert`. It also accepts a binary image (it is blurred first), but it is more faithful on the original micrograph, where crossing wires can still be told apart. The orientation scores are computed by tiles and cached, so memory stays at a few bytes per pixel even for very large mosaics; the M4 sample (2767 × 2424 px) takes about 20 s.

**Measurements.** Each wire node of `L` also has `width` and `truncated` (and `length_um`, `width_um` with `--pixel-size`). `--measurements wires.csv` saves one row per wire (id, degree, length, width, truncated) and the command line prints the medians.

* `length` (px) is measured along the center line, without the bridges added at the ends and cutting each tip where the brightness falls below half of the wire's contrast. On the synthetic images it is within 2–3 % of the true length (median) for wires recovered whole. A wire split in pieces by the tracer gives two shorter lengths, so in very dense areas the distribution is biased towards short wires.
* `truncated` marks wires with an end at the image border, an electrode, an excluded area or an empty mosaic margin: they may continue beyond what is seen, so their length is a lower bound. The printed length statistics leave them out.
* `width` (px) is the full width at half maximum of the intensity profile across the wire (median along it), measured only where no other wire is closer than 2 widths, so dense networks give it for a fraction of the wires. **In optical images of nanowires thinner than the resolution it is the width of the blurred image, not the diameter**: it tells the resolution and the focus, not the wire. It is meaningful as a diameter in SEM images where wires are many pixels wide.

**How faithful is it?** `benchmarks/` generates synthetic dark-field micrographs of curved wires with known positions (`synthetic.py`) and scores each method against them (`evaluate.py`). *Whole wires* is the fraction of the true wires that are recovered as a single node covering at least 80 % of their length; *nodes / wire* is the mean number of nodes on each recovered wire (1 is ideal); *spurious* are nodes that do not lie on a single wire (for wire tracing, mostly two wires chained together where one ends next to the other).

| 1024 × 1024 px, wires 3.2 px wide | Whole wires | Nodes / wire | Spurious nodes | Contacts precision / recall |
| --- | --- | --- | --- | --- |
| 150 wires, skeleton | 0–1 % | 1.8–1.9 | 9–15 % | 97–98 % / 28–49 %\* |
| 150 wires, wire tracing | 82–84 % | 1.02–1.08 | 3–8 % | 83–87 % / 68–82 % |
| 350 wires, skeleton | 0–1 % | 2.3–2.5 | 6–7 % | 96–97 % / 65–68 %\* |
| 350 wires, wire tracing | 69–70 % | 1.08–1.11 | 9–12 % | 87 % / 61–69 % |

\* For the skeleton method the contacts are counted after grouping its segments into wires *using the ground truth*, which the method itself cannot do; they are an upper bound. Its low recall in the sparse images comes from the cleaning, which prunes wires touching a single other wire.

```bash
python benchmarks/synthetic.py --out bench_data --n-wires 150 350 --seeds 0 1
python benchmarks/evaluate.py bench_data/*.png
```

Limitations: two wires crossing at a very shallow angle, or lying side by side over a long stretch, can still be swapped or chained into one node; bright clumps of many overlapping wires cannot be resolved at any method's resolution. Check the *Wires* view (or `plot_wires_on_image`) before trusting the numbers.

## Binarization and cleaning parameters

The defaults (`block_size=151, C=15, intensity_thresh=55`, opening of radius 1) were made for **optical dark-field** micrographs, where the background is close to black. They do **not** work for SEM images: the background is mid-gray, so `C=15` (threshold *below* the local mean) marks most of it as wire (in tests only 24 % of the detected "wire" pixels were real).

Starting points for SEM images of AgNW networks (Zeiss, 8192 × 6144 px tiles, wires 4–6 px wide at full resolution), available as presets in the GUI (tab 2):

| Preset | block size | C | intensity threshold | closing radius | opening radius | remove specks (px) |
| --- | --- | --- | --- | --- | --- | --- |
| Optical dark-field (default) | 151 | 15 | 55 | 0 | 1 | 0 |
| Optical dark-field, dense / thin wires | 31 | −8 | 0 | 1 | 0 | 10 |
| SEM, tiles downsampled 4× | 21 | −11 | 0 | 1 | 0 | 10 |
| SEM, tiles downsampled 2× | 61 | −23 | 0 | 2 | 0 | 10 |

From the command line: `--block-size 61 --C -23 --threshold 0 --closing-radius 2 --opening-radius 0 --min-object-size 10`.

How they were chosen and what to keep in mind:

* **Dense dark-field networks** (wires 1–2 px wide, bright clumps): the default preset also marks the halo around wires and crossings, merging neighboring wires into blobs, and its opening cuts the network apart. The *dense / thin wires* preset was chosen by visually comparing the skeleton with the image and by the connectivity of the cleaned network; adjust C between −6 (more continuous, more false contacts) and −10 (better separated, more fragmented). Bright clumps of overlapping wires cannot be resolved at this resolution.
* **Negative C** keeps only pixels clearly brighter than their surroundings.
* **No opening for thin wires.** The opening (radius 1) removes structures thinner than 3 px: on SEM mosaics downsampled 4× (wires 1–2 px) it erased 85–90 % of the wires, and 25–30 % at 2×.
* **A closing is needed.** In some regions the SEM shows the wires with bright edges and a darker core; the threshold then breaks them into fragments, and since `clean_network` prunes everything that is not part of a cycle, the graph falls apart (a binarization with 98 % pixel agreement gave 6 nodes instead of ~4000). A closing of radius 1–2 bridges those gaps.
* The values were chosen by comparing the resulting **graph** (number of nodes after cleaning) with the graph of a reference segmentation made at full resolution and reduced to the same scale, on tiles from a dark and a bright region: 0.76–0.99 of the reference nodes at 4×, 0.98–1.03 at 2×.
* **Resolution changes the topology in dense regions.** Compared with the full-resolution reference, the graph at 4× had about the same number of nodes in a sparse region but ~3× more in a dense region with many nearly parallel wires (they touch at low resolution and create spurious junctions); at 2× it was 1.2–1.9×. For quantitative results in dense areas, analyze at the highest resolution you can, e.g. by regions.
* The graph is very sensitive to these settings: always check the binary layer and the graph visually before trusting the numbers.

## Electrodes from a later photo (`electrode_registration.py`)

Typical case: the network is mapped first (e.g. a 10× mosaic) and the electrodes are deposited afterwards; a new photo shows them, usually at another magnification (e.g. 5×), slightly rotated and shifted. This module transfers them to the mosaic:

1. **Registration**: the same features of the wire network are found in both images (SIFT) and the similarity transform (scale + rotation + translation) from the photo to the image is fitted with RANSAC. The scale can be given or searched automatically.
2. **Detection**: the two electrodes are found in the photo by a smoothed feature (`warm` = red minus blue, `bright`, `dark`, or `auto`, which picks the one that best separates two classes); the two largest regions are kept, ordered left→right (or top→bottom): the first is Vin, the second Vout.
3. **Transfer**: both masks are warped to the image coordinates.

```bash
# writes JTA_Ag_O3_electrode_mask1.tif (Vin), _mask2.tif (Vout) and _electrode_check.jpg next to the image
python electrode_registration.py JTA_Ag_O3.tif JTA_Ag_O3_electrodes.JPG --scale 2
python electrode_registration.py -h
```

In the GUI (tab 3, *From a later photo*): choose the photo, the scale (image px per photo px, e.g. 2 for a 5× photo on a 10× mosaic, or *auto*) and how the electrodes look, then *Align photo & detect electrodes*. The electrodes layer is filled and the aligned photo is added as a semi-transparent layer to check them; touch them up with the drawing tools and *Save masks...*.

From Python: `electrodes_from_photo(image, photo, scale=None, feature="auto", sigma=15)` returns `(mask1, mask2, info)`; `info` holds the transform `M`, `scale`, `rotation_deg`, the number of consistent matches, the residual and warnings. Lower-level pieces: `register`, `detect_electrodes`, `warp_mask`, `warp_photo`, `check_image`.

Notes:

* The registration needs the photo to show the same network (a few hundred consistent matches are typical; at least 25 are required). On JTA_Ag_O3 (5× photo on a 10× mosaic) it found scale 1.987 and rotation −2.37° from ~2500 matches, with a median residual of 0.56 px; in synthetic tests it recovered any rotation and magnification ratios up to 5× within ~1 px.
* The detection is a starting point. If an electrode touches the border of the photo you get a warning: it may continue outside the photo, or the detection may include some bare network there. Always check the edges facing the gap, which define the contacts.
* Under dark field the same wire can look brighter or dimmer depending on its orientation and the illumination, so the photo and the mosaic do not show exactly the same wires; this does not affect the registration.

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

Wire tracing (nodes = whole wires):

```python
from WireTracing import WireAnalysis
from visualization import plot_wires_on_image

WA = WireAnalysis("path/M4.tif")          # wire width estimated; or wire_width=3.5
WA.complete_analysis(
    electrodes_path=("example/M4/M4_electrode_mask1.tif", "example/M4/M4_electrode_mask2.tif"),
    save_graph_path="M4_wires.graphml",
    verbose=True,
)
WA.wires   # list of (n, 2) arrays: the center line (x, y) of every wire found
WA.L       # nodes = wires (index into WA.wires) and Vin_i / Vout_i; edges = contacts
WA.G       # spatial graph: nodes = contacts, edges = pieces of wire between them

plot_wires_on_image(WA, xlim=(1000, 1500), ylim=(1500, 1000))
```

Step by step, instead of `complete_analysis`:

```python
WA.load_electrodes(path_vin, path_vout)   # optional, first
WA.orientation_scores()
WA.trace()
WA.join()
WA.extend_ends()
WA.find_contacts()
WA.build_graphs()                         # drop_dangling=False keeps every wire
WA.save_graph("output.graphml", save_line_graph=True)
```

Skeleton method:

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
* **`img_cleaning(hole_size=2, opening_radius=1, min_object_size=0, closing_radius=0)`**: optional closing (bridges small gaps), fills small holes, applies an opening with a disk of `opening_radius` (0 = none) and removes objects of up to `min_object_size` px. The defaults reproduce the original behavior (holes + opening of radius 1).
* **`skeletonize()`**: reduces structures to 1-pixel width; if there are electrodes, removes them from the skeleton and keeps dilated masks to connect them.
* **`get_network()`**: converts the skeleton into a graph (`skan` → networkx) and adds `Vin_i` / `Vout_i` nodes where the skeleton touches the electrodes.
* **`clean_network()`**: iteratively removes isolated nodes, prunes degree-1 nodes, collapses degree-2 nodes and removes edges between electrodes.
* **`convert_to_line_graph()`**: builds `SA.L`, the line graph of `SA.G`.
* **`save_graph(path, save_line_graph=True)`**: exports to `.graphml`. When saving the spatial graph `G`, the non-scalar attributes added by skan (pixel path of each branch, image shape) are dropped, since GraphML cannot store them, and each node gets its pixel coordinates as `x` / `y` attributes (electrode nodes: their contact point); the image size is stored as `image_height` / `image_width`.
* **`plot_graph(save_path=None)`**: quick plot of the graph over the image.
* **`complete_analysis(clean_network=True, electrodes_path=None, save_graph_path=None, plot=False, save_plot_path=None, verbose=False, opening_radius=1, min_object_size=0, closing_radius=0)`**: runs the whole pipeline.

### `WireTracing`

* **`WireAnalysis(img_path=None, img=None, wire_width=None, invert=False)`**: loads the micrograph (path or array, gray or color). `wire_width=None` measures it with `estimate_wire_width`.
* **`load_electrodes(path1, path2)`**: Vin and Vout masks (paths or arrays); no wire is traced inside them.
* **`load_exclusion(mask)`**: areas to ignore (path or array, nonzero = ignore).
* **`orientation_scores(n_orientations=16, sigma_across=None, sigma_along=None, tile_size=256, max_cache_mb=1024)`**: sets up the ridge responses per orientation (computed by tiles on demand) and the best response per pixel (`R_max`, `R_arg`).
* **`trace(seed_threshold=None, low_threshold=None, step=1.5, max_turn_deg=4, max_gap=2, ...)`**: follows every wire; `raw_wires` holds the traced pieces.
* **`join(max_distance=None, max_angle_deg=20)`**: joins pieces of the same wire; result in `wires`.
* **`extend_ends(max_extension=None, min_response=None)`**: closes the small gap between a wire's end and the wire it runs into.
* **`find_contacts(contact_distance=None, end_tolerance=None)`**: `contacts` (i, j, x, y) and `electrode_contacts`.
* **`measure_wires(step=2.0, pixel_size=None)`**: `measurements`, one dict per wire with `length`, `width`, `truncated`; `measurement_table(path)` writes them as CSV and `measurement_summary()` gives medians and quartiles.
* **`build_graphs(drop_dangling=True)`**: builds `L` (and `L_full` before cleaning) and `G`.
* **`save_graph(path, save_line_graph=True)`**: exports `L` or `G` to `.graphml` (with `x`/`y` on every node).
* **`complete_analysis(electrodes_path=None, clean_network=True, save_graph_path=None, ...)`**: runs everything.
* **`wire_label_image(thickness=None, wires=None)`**: image with each wire drawn as its index + 1.
* **`estimate_wire_width(img)`**: typical wire width in px (FWHM of the profile across the strongest ridges; distance transform for binary images).

### `visualization.py`

* **`plot_network_on_image(SA, xlim=None, ylim=None, background="skeleton", ...)`**: draws `SA.G` over the image (Vin in red, Vout in blue). `background` can be `"skeleton"`, `"img"`, `"raw"`, `"original"`, a path or an array. `xlim`/`ylim` zoom in and scale nodes and edges accordingly.
* **`plot_line_graph_on_image(SA, ...)`**: same, for the line graph (each node at the centroid of its wire).
* **`get_node_positions(SA)`** / **`get_line_graph_positions(SA)`**: `(x, y)` node positions for plotting with networkx.
* **`plot_wires_on_image(WA, ...)`**: each wire of a `WireAnalysis` in its own color over the micrograph, with contacts (white) and electrode contacts (red / blue).

All plotting functions accept either a `SkeletonAnalysis` or a `WireAnalysis`.

## Notes

* The stitching was checked against a mosaic made by hand in Fiji (sample M4, 6 tiles): `stitch(...)` with default parameters gives the same 2767 × 2424 image (mean difference 0.3 gray levels out of 255).
* `SkeletonAnalysis` accepts the binary image and the electrode masks either as file paths or as arrays (`SkeletonAnalysis(img=mask)`, `load_electrodes(mask1, mask2)`).
* The project uses `opencv-python-headless` (OpenCV without its own windows), which avoids Qt conflicts with napari. If you already had `opencv-python` installed, uninstall it first: `pip uninstall opencv-python`.
* The input image to `SkeletonAnalysis` must be binarized (wires in white), preferably as `.tif`.
* Compatible with scikit-image both before and after 0.26 (change from `min_size`/`area_threshold` to `max_size`).
* With the example sample, `python main.py --method skeleton` reproduces the thesis' `M4_effective.graphml` (6703 nodes and 13242 edges in the line graph). On the original M4 micrograph (`muestras/M4/M4.tif` in the thesis repository) with the same electrodes, wire tracing gives about 4900 wires and 16900 contacts; most of the difference with earlier versions are false traces over the fields of dust outside the sample, now discarded.

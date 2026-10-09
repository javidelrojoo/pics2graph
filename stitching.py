"""
stitching
=========

Stitch a set of overlapping microscope tiles into a single mosaic by running
Fiji's **Grid/Collection stitching** plugin (Preibisch et al., 2009) in
headless mode from Python.

Nothing is reimplemented here: Python writes an ImageJ macro with the chosen
parameters, launches your local Fiji installation without GUI, waits for it to
finish and collects the fused image. The result is the same as running
*Plugins > Stitching > Grid/Collection stitching* by hand.

Requirements
------------
A local Fiji installation (https://fiji.sc) with the Stitching plugin, which is
included by default. Fiji is located, in this order, from:
    1. the ``fiji_path`` argument (the ``Fiji.app`` folder or the launcher),
    2. the ``FIJI_PATH`` environment variable,
    3. common install locations (``~/Fiji.app``, ``C:/Fiji.app``,
       ``C:/fiji-*/Fiji.app``, ``~/Downloads/fiji-*/Fiji.app``,
       ``/Applications/Fiji.app``, ...).

Large tile sets (e.g. SEM)
--------------------------
ImageJ cannot hold a single image plane larger than 2^31 - 1 pixels (~2147
Mpx), and the fusion needs the whole mosaic in memory. For big sets, reduce
the tiles before stitching with ``downsample`` (an integer factor; each n x n
block of pixels is averaged) and convert them to 8-bit gray with
``tiles_to_gray``. ``stitch`` estimates the mosaic size beforehand and stops
with a suggested factor if it cannot fit.

Example
-------
    from stitching import stitch
    mosaic_path = stitch("tiles/M4", output_path="M4/M4.tif")

    # 180 SEM tiles of 8192 x 6144 px: reduce 4x and convert to 8-bit gray
    stitch("tiles/SEM", downsample=4, tiles_to_gray=True)

    # Grid acquisition (snake by rows, 2 x 3 tiles, ~30 % overlap)
    stitch("tiles/M4", mode="grid", grid_type="snake-by-rows", order="right-down",
           grid_size=(2, 3), tile_overlap=30, file_names="Image_{iiiii}.tif",
           first_index=30938)

Command line
------------
    python stitching.py tiles/M4 -o M4/M4.tif
    python stitching.py -h
"""

import argparse
import glob
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")

# Exact option strings expected by the plugin (note the trailing spaces in some
# of them: they are part of the plugin's choice lists).
GRID_TYPES = {
    "row-by-row": "Grid: row-by-row",
    "column-by-column": "Grid: column-by-column",
    "snake-by-rows": "Grid: snake by rows",
    "snake-by-columns": "Grid: snake by columns",
}
ROW_ORDERS = {
    "right-down": "Right & Down                ",
    "left-down": "Left & Down",
    "right-up": "Right & Up",
    "left-up": "Left & Up",
}
COLUMN_ORDERS = {
    "down-right": "Down & Right                ",
    "down-left": "Down & Left",
    "up-right": "Up & Right",
    "up-left": "Up & Left",
}
FUSION_METHODS = (
    "Linear Blending",
    "Average",
    "Median",
    "Max. Intensity",
    "Min. Intensity",
    "Intensity of random input tile",
)
COMPUTATION = {
    "time": "Save computation time (but use more RAM)",
    "memory": "Save memory (but be slower)",
}
# computation="auto" uses "memory" when the tiles add up to more than this:
AUTO_MEMORY_BYTES = 2 * 1024 ** 3

IMAGEJ_MAX_PIXELS = 2 ** 31 - 1     # largest image plane ImageJ can hold
# Fraction of the summed tile area that ends up in the mosaic (assumes at least
# ~10 % overlap in each direction). Used only for the size estimate.
ASSUMED_COVERAGE = 0.8
TEMP_PREFIX = "pics2graph_stitch_"


# =============================================================================
#  Locating Fiji
# =============================================================================
def _launcher_candidates():
    """Launcher executables to look for inside a Fiji.app folder, by OS."""
    system = platform.system()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    if system == "Windows":
        return [f"fiji-windows-{arch}.exe", "ImageJ-win64.exe"]
    if system == "Darwin":
        return [f"fiji-macos-{arch}", "fiji-macos-universal",
                f"Contents/MacOS/fiji-macos-{arch}", "Contents/MacOS/ImageJ-macosx"]
    return [f"fiji-linux-{arch}", "ImageJ-linux64"]


def _launcher_in(folder):
    for name in _launcher_candidates():
        cand = os.path.join(folder, name)
        if os.path.isfile(cand):
            return cand
    return None


def find_fiji(fiji_path=None):
    """
    Return the path to the Fiji launcher executable.

    ``fiji_path`` can be the launcher itself or the ``Fiji.app`` folder (or a
    folder that contains ``Fiji.app``). If not given, ``FIJI_PATH`` and common
    install locations are tried. Raises ``FileNotFoundError`` if not found.
    """
    tried = []
    candidates = [fiji_path, os.environ.get("FIJI_PATH")]
    if not any(candidates):
        home = os.path.expanduser("~")
        patterns = [
            os.path.join(home, "Fiji.app"),
            os.path.join(home, "Fiji"),
            os.path.join(home, "*", "Fiji.app"),
            os.path.join(home, "Downloads", "fiji*", "Fiji.app"),
            os.path.join(home, "Desktop", "fiji*", "Fiji.app"),
            "C:/Fiji.app",
            "C:/fiji*/Fiji.app",
            "C:/Program Files/Fiji.app",
            "C:/Program Files/fiji*/Fiji.app",
            "/Applications/Fiji.app",
            "/opt/Fiji.app",
            "/opt/fiji*/Fiji.app",
        ]
        for pat in patterns:
            candidates.extend(sorted(glob.glob(pat)))

    for cand in candidates:
        if not cand:
            continue
        cand = os.path.expanduser(cand)
        tried.append(cand)
        if os.path.isfile(cand):
            return cand
        if os.path.isdir(cand):
            exe = _launcher_in(cand) or _launcher_in(os.path.join(cand, "Fiji.app"))
            if exe:
                return exe

    raise FileNotFoundError(
        "Fiji was not found. Pass fiji_path='.../Fiji.app' (or --fiji on the "
        "command line), or set the FIJI_PATH environment variable.\n"
        "Tried: " + (", ".join(tried) if tried else "(nothing)")
    )


# =============================================================================
#  Building the macro
# =============================================================================
def _macro_path(path):
    """Path in the form expected inside a macro: absolute, forward slashes."""
    path = os.path.abspath(path).replace("\\", "/")
    if "]" in path or '"' in path:
        raise ValueError(f"Paths with ']' or '\"' cannot be passed to the plugin: {path}")
    return path


def _fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def build_options(directory, output_directory, mode="unknown",
                  grid_type="snake-by-rows", order="right-down", grid_size=None,
                  tile_overlap=20, first_index=1, file_names=None,
                  layout_file="TileConfiguration.txt",
                  fusion_method="Linear Blending", regression_threshold=0.30,
                  max_avg_displacement_threshold=2.50,
                  absolute_displacement_threshold=3.50, compute_overlap=True,
                  subpixel_accuracy=True, computation="time", extra_options=""):
    """
    Build the option string for ``run("Grid/Collection stitching", ...)``.

    See ``stitch`` for the meaning of each parameter.
    """
    directory = _macro_path(directory)
    output_directory = _macro_path(output_directory)

    if fusion_method not in FUSION_METHODS:
        raise ValueError(f"fusion_method must be one of {FUSION_METHODS}")
    if computation not in COMPUTATION:
        raise ValueError(f"computation must be one of {tuple(COMPUTATION)} (or 'auto' in stitch)")

    if mode == "unknown":
        opts = ["type=[Unknown position]", "order=[All files in directory]",
                f"directory=[{directory}]",
                "output_textfile_name=TileConfiguration.txt"]
    elif mode == "grid":
        if grid_type not in GRID_TYPES:
            raise ValueError(f"grid_type must be one of {tuple(GRID_TYPES)}")
        orders = COLUMN_ORDERS if "column" in grid_type else ROW_ORDERS
        if order not in orders:
            raise ValueError(f"order for '{grid_type}' must be one of {tuple(orders)}")
        if not grid_size or len(grid_size) != 2:
            raise ValueError("grid mode needs grid_size=(n_columns, n_rows)")
        if not file_names:
            raise ValueError("grid mode needs file_names, e.g. 'tile_{ii}.tif' "
                             "({i...} is replaced by the tile index)")
        opts = [f"type=[{GRID_TYPES[grid_type]}]", f"order=[{orders[order]}]",
                f"grid_size_x={int(grid_size[0])}", f"grid_size_y={int(grid_size[1])}",
                f"tile_overlap={_fmt(tile_overlap)}",
                f"first_file_index_i={int(first_index)}",
                f"directory=[{directory}]", f"file_names=[{file_names}]",
                "output_textfile_name=TileConfiguration.txt"]
    elif mode == "positions":
        opts = ["type=[Positions from file]", "order=[Defined by TileConfiguration]",
                f"directory=[{directory}]",
                f"layout_file=[{os.path.basename(layout_file)}]"]
    else:
        raise ValueError("mode must be 'unknown', 'grid' or 'positions'")

    opts += [f"fusion_method=[{fusion_method}]",
             f"regression_threshold={_fmt(float(regression_threshold))}",
             f"max/avg_displacement_threshold={_fmt(float(max_avg_displacement_threshold))}",
             f"absolute_displacement_threshold={_fmt(float(absolute_displacement_threshold))}"]
    if compute_overlap and mode != "unknown":   # always computed in 'unknown' mode
        opts.append("compute_overlap")
    if subpixel_accuracy:
        opts.append("subpixel_accuracy")
    opts += [f"computation_parameters=[{COMPUTATION[computation]}]",
             "image_output=[Write to disk]",
             f"output_directory=[{output_directory}]"]
    if extra_options:
        opts.append(extra_options.strip())
    return " ".join(opts)


def build_macro(options, done_marker):
    """ImageJ macro that runs the plugin and writes ``done_marker`` at the end."""
    options = options.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'run("Grid/Collection stitching", "{options}");\n'
        f'File.saveString("done", "{_macro_path(done_marker)}");\n'
    )


# =============================================================================
#  Collecting the result
# =============================================================================
def _read_fused(output_directory):
    """
    Read the plugin's output (``img_t1_z1_c1``, ``..._c2``, ...) and return it as
    a single array: 2-D for one channel, (H, W, 3) in BGR order for RGB (ready
    for ``cv2.imwrite``).
    """
    files = sorted(glob.glob(os.path.join(output_directory, "img_t1_z1_c*")),
                   key=lambda p: int(p.rsplit("_c", 1)[1].split(".")[0]))
    if not files:
        raise RuntimeError(f"Fiji did not write any fused image in {output_directory}.")
    channels = []
    for f in files:
        ch = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if ch is None:
            raise RuntimeError(f"Could not read Fiji's output: {f}")
        if ch.ndim == 3:            # already multichannel
            return ch
        channels.append(ch)
    if len(channels) == 1:
        return channels[0]
    if len(channels) == 3:
        return np.dstack(channels[::-1])     # RGB -> BGR for OpenCV
    return np.dstack(channels)


def list_tiles(tiles_dir, pattern="*", exclude=()):
    """
    Image files in ``tiles_dir`` matching ``pattern``, sorted by name.
    Files named ``Fused*`` / ``stitched*`` and images whose size differs from
    the majority of the tiles are skipped.
    """
    exclude = {os.path.normcase(os.path.abspath(e)) for e in exclude if e}
    tiles = []
    for p in sorted(glob.glob(os.path.join(tiles_dir, pattern))):
        if not os.path.isfile(p) or not p.lower().endswith(IMAGE_EXTENSIONS):
            continue
        if os.path.normcase(os.path.abspath(p)) in exclude:
            continue
        if os.path.basename(p).lower().startswith(("fused", "stitched")):
            continue                          # results of previous runs
        tiles.append(p)
    return _same_size_only(tiles)


def _image_size(path):
    """(width, height) read from the file header, without loading the image."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return im.size


def _same_size_only(tiles):
    """
    Keep only the tiles with the most common size. Images of a different size
    (e.g. a mosaic from a previous run saved in the same folder) are skipped
    with a warning.
    """
    if len(tiles) < 2:
        return tiles
    sizes = {}
    for t in tiles:
        try:
            sizes[t] = _image_size(t)
        except Exception:
            sizes[t] = None
    valid = [s for s in sizes.values() if s is not None]
    if not valid:
        return tiles
    common = max(set(valid), key=valid.count)
    kept = [t for t in tiles if sizes[t] == common]
    for t in tiles:
        if sizes[t] != common:
            print(f"[warning] skipping {os.path.basename(t)}: size {sizes[t]} "
                  f"differs from the tiles' size {common}.")
    return kept


# =============================================================================
#  Preparing the tiles (link / copy / reduce) and checking the mosaic size
# =============================================================================
def _palette_is_gray(im):
    """
    True if a palette ('P') image only *uses* gray colors (R == G == B).
    Unused entries are ignored: SEM software (e.g. Zeiss) reserves a few
    colored entries for annotations that most images never use.
    """
    pal = im.getpalette() or []
    counts = im.histogram()
    for i, n in enumerate(counts):
        if n and 3 * i + 2 < len(pal) and not (pal[3 * i] == pal[3 * i + 1] == pal[3 * i + 2]):
            return False
    return True


def _tile_info(path):
    """(width, height, channels, bytes_per_channel, PIL mode) of an image file."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        mode = im.mode
        w, h = im.size
        if mode == "P":
            mode = "P-gray" if _palette_is_gray(im) else "P"
    channels = {"RGB": 3, "RGBA": 4, "CMYK": 4, "LA": 2, "P": 3}.get(mode, 1)
    depth = 2 if mode.startswith("I;16") or mode == "I" else (4 if mode == "F" else 1)
    return w, h, channels, depth, mode


def _needs_rewrite(info, downsample, tiles_to_gray):
    """Whether the tiles must be rewritten (reduced / converted) for Fiji."""
    w, h, channels, depth, mode = info
    return (downsample > 1 or mode.startswith("P")
            or (tiles_to_gray and (channels > 1 or depth > 1)))


def _prepare_tile(src, dst, downsample, tiles_to_gray):
    """Write ``src`` to ``dst`` reduced by ``downsample`` and/or in 8-bit gray."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(src) as im:
        if im.mode == "P":                      # palette (e.g. SEM): expand the LUT
            im = im.convert("L" if tiles_to_gray or _palette_is_gray(im) else "RGB")
        if tiles_to_gray and im.mode != "L":
            if im.mode.startswith("I") or im.mode == "F":
                import numpy as np
                a = np.asarray(im, dtype=np.float64)
                lo, hi = np.percentile(a, (0.01, 99.99))
                a = np.clip((a - lo) * (255.0 / max(hi - lo, 1e-12)), 0, 255)
                im = Image.fromarray(a.astype(np.uint8))
            else:
                im = im.convert("L")
        if downsample > 1:
            im = im.reduce(int(downsample))     # mean of each n x n block
        ext = os.path.splitext(dst)[1].lower()
        if ext in (".jpg", ".jpeg"):
            im.save(dst, quality=95)
        else:
            im.save(dst)


def _link_or_copy(src, dst):
    """Hard-link ``src`` to ``dst`` (no extra disk space); copy if not possible."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _scale_layout(src, dst, factor):
    """Copy a TileConfiguration file dividing its coordinates by ``factor``."""
    number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    pattern = re.compile(r"\(\s*(" + number + r")\s*,\s*(" + number + r")\s*(?:,\s*(" + number + r"))?\s*\)")

    def scale(m):
        vals = [float(v) / factor for v in m.groups() if v is not None]
        return "(" + ", ".join(repr(v) for v in vals) + ")"

    with open(src, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    with open(dst, "w", encoding="utf-8") as fh:
        for line in lines:
            if ";" in line and not line.lstrip().startswith("#"):
                line = pattern.sub(scale, line)
            fh.write(line)


def estimate_mosaic(tiles, downsample=1, tiles_to_gray=False):
    """
    Rough size of the mosaic that ``stitch`` would produce.

    Returns a dict with the size of one prepared tile, the estimated mosaic
    pixels (summed tile area x ``ASSUMED_COVERAGE``), the bytes of all tiles,
    and the smallest ``downsample`` that keeps the mosaic under ImageJ's limit.
    """
    w, h, channels, depth, mode = _tile_info(tiles[0])
    if tiles_to_gray:
        channels, depth = 1, 1
    tw, th = w // downsample, h // downsample
    tile_px = tw * th
    mosaic_px = len(tiles) * tile_px * ASSUMED_COVERAGE
    full_px = len(tiles) * w * h * ASSUMED_COVERAGE
    min_factor = max(1, math.ceil(math.sqrt(full_px / IMAGEJ_MAX_PIXELS)))
    comfortable = max(1, math.ceil(math.sqrt(full_px / 600e6)))   # ~0.6 Gpx
    return dict(tile_size=(tw, th), channels=channels, bytes_per_channel=depth,
                mosaic_pixels=mosaic_px,
                mosaic_bytes=mosaic_px * channels * depth,
                tiles_bytes=len(tiles) * tile_px * channels * depth,
                min_downsample=min_factor, suggested_downsample=comfortable)


def _cleanup_stale_temp(max_age_hours=6):
    """Remove working folders left by interrupted runs (older than a few hours)."""
    now = time.time()
    for d in glob.glob(os.path.join(tempfile.gettempdir(), TEMP_PREFIX + "*")):
        try:
            if os.path.isdir(d) and now - os.path.getmtime(d) > max_age_hours * 3600:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


# =============================================================================
#  Main function
# =============================================================================
def stitch(tiles_dir, output_path=None, mode="unknown", pattern="*", tiles=None,
           grid_type="snake-by-rows", order="right-down", grid_size=None,
           tile_overlap=20, first_index=1, file_names=None,
           layout_file=None, fusion_method="Linear Blending",
           regression_threshold=0.30, max_avg_displacement_threshold=2.50,
           absolute_displacement_threshold=3.50, compute_overlap=True,
           subpixel_accuracy=True, computation="auto", extra_options="",
           grayscale=False, downsample=1, tiles_to_gray=False, check_size=True,
           fiji_path=None, memory=None, timeout=None, keep_temp=False,
           verbose=True):
    """
    Stitch the tiles in ``tiles_dir`` with Fiji and save the fused mosaic.

    The tiles are placed in a temporary folder before running Fiji (hard
    links when possible, so no extra disk space; reduced copies when
    ``downsample`` or ``tiles_to_gray`` apply), so the original folder is not
    modified (the plugin writes its TileConfiguration files next to the tiles)
    and non-image files are ignored.

    Parameters
    ----------
    tiles_dir : str
        Folder with the tiles.
    output_path : str, optional
        Where to save the mosaic. Defaults to ``<tiles_dir>/Fused.tif``.
        The registered tile positions are saved next to it as
        ``<name>_TileConfiguration.registered.txt``.
    mode : {"unknown", "grid", "positions"}
        "unknown"   : *Unknown position* - the plugin finds the arrangement by
                      itself (all tiles against all tiles). Simplest option.
        "grid"      : *Grid: ...* - tiles acquired in a regular grid; requires
                      ``grid_type``, ``order``, ``grid_size``, ``tile_overlap``,
                      ``file_names`` and ``first_index``.
        "positions" : *Positions from file* - approximate positions given in
                      a TileConfiguration file (``layout_file``).
    pattern : str
        Glob pattern to select the tiles inside ``tiles_dir`` (e.g. "*.JPG").
    tiles : list of str, optional
        Explicit list of tile files (overrides ``tiles_dir``/``pattern``).
    grid_type : {"row-by-row", "column-by-column", "snake-by-rows", "snake-by-columns"}
    order : str
        For rows: "right-down", "left-down", "right-up", "left-up".
        For columns: "down-right", "down-left", "up-right", "up-left".
    grid_size : (int, int)
        Number of tiles (columns, rows).
    tile_overlap : float
        Approximate overlap between neighboring tiles, in percent.
    first_index : int
        Index of the first tile in ``file_names``.
    file_names : str
        File name pattern with ``{i}`` / ``{ii}`` / ``{iii}``... for the tile
        index (e.g. "Image_{iiiii}.tif").
    layout_file : str
        TileConfiguration file for ``mode="positions"``.
    fusion_method : str
        One of ``FUSION_METHODS`` ("Linear Blending" by default).
    regression_threshold, max_avg_displacement_threshold, absolute_displacement_threshold : float
        Thresholds of the global optimization (plugin defaults: 0.30, 2.50, 3.50).
    compute_overlap : bool
        In grid/positions modes, refine the positions by phase correlation.
    subpixel_accuracy : bool
        Subpixel registration.
    computation : {"auto", "time", "memory"}
        Trade-off between speed and RAM. "auto" uses "memory" when the tiles
        add up to more than 2 GB.
    extra_options : str
        Extra options appended verbatim to the plugin call (for anything not
        covered above; record it with Fiji's macro recorder).
    grayscale : bool
        Save the mosaic in grayscale instead of color.
    downsample : int
        Reduce each tile by this factor before stitching (each n x n block of
        pixels is averaged). The mosaic shrinks by n^2 in pixels and memory;
        the registered positions refer to the reduced tiles.
    tiles_to_gray : bool
        Convert the tiles to 8-bit grayscale before stitching (3x less memory
        for color tiles, 2x for 16-bit). Palette images (common in SEM) are
        always expanded through their lookup table.
    check_size : bool
        Estimate the mosaic size first and stop with a suggested ``downsample``
        if it exceeds what ImageJ can hold.
    fiji_path : str, optional
        Fiji.app folder or launcher. See ``find_fiji``.
    memory : str, optional
        Maximum Java heap for Fiji (e.g. "8g", "75%").
    timeout : float, optional
        Seconds before giving up.
    keep_temp : bool
        Keep the temporary working folder (for debugging).
    verbose : bool
        Show Fiji's output and progress messages.

    Returns
    -------
    str : path of the saved mosaic.
    """
    if output_path is None:
        output_path = os.path.join(tiles_dir, "Fused.tif")
    output_path = os.path.abspath(output_path)

    if tiles is None:
        tiles = list_tiles(tiles_dir, pattern, exclude=(output_path,))
    if len(tiles) < 2:
        raise ValueError(f"At least 2 tiles are needed; found {len(tiles)} in {tiles_dir}.")
    downsample = int(downsample)
    if downsample < 1:
        raise ValueError("downsample must be an integer >= 1")

    est = estimate_mosaic(tiles, downsample, tiles_to_gray)
    if verbose:
        tw, th = est["tile_size"]
        print(f"{len(tiles)} tiles of {tw} x {th} px"
              f"{' (after reducing ' + str(downsample) + 'x)' if downsample > 1 else ''}; "
              f"estimated mosaic ~{est['mosaic_pixels'] / 1e6:,.0f} Mpx "
              f"(~{est['mosaic_bytes'] / 1024 ** 3:.1f} GB).")
    if check_size and est["mosaic_pixels"] > IMAGEJ_MAX_PIXELS:
        raise ValueError(
            f"The mosaic would have ~{est['mosaic_pixels'] / 1e6:,.0f} Mpx, more than "
            f"ImageJ can hold in one image ({IMAGEJ_MAX_PIXELS / 1e6:,.0f} Mpx). Reduce the "
            f"tiles with downsample={max(est['min_downsample'], downsample + 1)} or more "
            f"(downsample={est['suggested_downsample']} keeps the later analysis "
            f"comfortable on ~16 GB of RAM), and consider tiles_to_gray=True. "
            f"Use check_size=False to try anyway.")
    fiji = find_fiji(fiji_path)
    if computation == "auto":
        computation = "memory" if est["tiles_bytes"] > AUTO_MEMORY_BYTES else "time"
        if verbose and computation == "memory":
            print("Large tile set: using Fiji's low-memory mode (slower).")

    _cleanup_stale_temp()
    work = tempfile.mkdtemp(prefix=TEMP_PREFIX)
    in_dir = os.path.join(work, "tiles")
    out_dir = os.path.join(work, "fused")
    os.makedirs(in_dir)
    os.makedirs(out_dir)
    try:
        rewrite = _needs_rewrite(_tile_info(tiles[0]), downsample, tiles_to_gray)
        t_prep = time.time()
        for i, t in enumerate(tiles, 1):
            dst = os.path.join(in_dir, os.path.basename(t))
            if rewrite:
                _prepare_tile(t, dst, downsample, tiles_to_gray)
            else:
                _link_or_copy(t, dst)
            if verbose and rewrite and (i % 10 == 0 or i == len(tiles)):
                print(f"Preparing tiles: {i}/{len(tiles)} ({time.time() - t_prep:.0f} s)",
                      flush=True)
        if mode == "positions":
            if layout_file is None:
                layout_file = os.path.join(tiles_dir, "TileConfiguration.txt")
            _scale_layout(layout_file, os.path.join(in_dir, os.path.basename(layout_file)),
                          downsample)

        options = build_options(
            in_dir, out_dir, mode=mode, grid_type=grid_type, order=order,
            grid_size=grid_size, tile_overlap=tile_overlap, first_index=first_index,
            file_names=file_names, layout_file=layout_file or "TileConfiguration.txt",
            fusion_method=fusion_method, regression_threshold=regression_threshold,
            max_avg_displacement_threshold=max_avg_displacement_threshold,
            absolute_displacement_threshold=absolute_displacement_threshold,
            compute_overlap=compute_overlap, subpixel_accuracy=subpixel_accuracy,
            computation=computation, extra_options=extra_options)
        done_marker = os.path.join(work, "done.txt")
        macro = os.path.join(work, "stitch.ijm")
        with open(macro, "w", encoding="utf-8") as fh:
            fh.write(build_macro(options, done_marker))

        # In headless mode the launcher already passes "-batch" to ImageJ, so
        # the macro file goes right after it (not with "-macro").
        cmd = [fiji, "--headless"]
        if memory:
            cmd.append(f"--heap={memory}")
        if verbose:
            cmd.append("--console")
        cmd.append(macro)

        if verbose:
            print(f"Stitching {len(tiles)} tiles with Fiji ({fiji}) ...")
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=None if verbose else subprocess.DEVNULL,
            timeout=timeout,
        )
        if not os.path.exists(done_marker):
            raise RuntimeError(
                f"Fiji's stitching did not finish (exit code {proc.returncode}). "
                f"Run with verbose=True to see Fiji's messages; the macro that was "
                f"run is in {macro} (use keep_temp=True to keep it)."
            )

        fused = _read_fused(out_dir)
        if grayscale and fused.ndim == 3:
            fused = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if not cv2.imwrite(output_path, fused):
            raise RuntimeError(f"Could not write {output_path}")

        registered = os.path.join(in_dir, "TileConfiguration.registered.txt")
        if os.path.exists(registered):
            stem = os.path.splitext(output_path)[0]
            shutil.copy2(registered, f"{stem}_TileConfiguration.registered.txt")

        if verbose:
            h, w = fused.shape[:2]
            print(f"Mosaic of {w} x {h} px saved to {output_path} "
                  f"({time.time() - t0:.0f} s).")
        return output_path
    finally:
        if keep_temp:
            if verbose:
                print(f"Temporary folder kept at {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


# =============================================================================
#  Command line
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(
        description="Stitch microscope tiles into one mosaic using Fiji's "
                    "Grid/Collection stitching plugin (headless).")
    p.add_argument("tiles_dir", help="Folder with the tiles.")
    p.add_argument("-o", "--output", default=None,
                   help="Output mosaic (default: <tiles_dir>/Fused.tif).")
    p.add_argument("--fiji", default=None, help="Fiji.app folder or launcher.")
    p.add_argument("--pattern", default="*", help="Glob to select tiles (e.g. '*.JPG').")
    p.add_argument("--mode", choices=("unknown", "grid", "positions"), default="unknown")
    p.add_argument("--grid-type", choices=tuple(GRID_TYPES), default="snake-by-rows")
    p.add_argument("--order", default="right-down",
                   help="right-down, left-down, right-up, left-up (rows) or "
                        "down-right, down-left, up-right, up-left (columns).")
    p.add_argument("--grid-size", nargs=2, type=int, metavar=("COLS", "ROWS"))
    p.add_argument("--overlap", type=float, default=20, help="Tile overlap in %%.")
    p.add_argument("--first-index", type=int, default=1)
    p.add_argument("--file-names", default=None, help="e.g. 'Image_{iiiii}.tif'.")
    p.add_argument("--layout-file", default=None, help="TileConfiguration for --mode positions.")
    p.add_argument("--fusion", choices=FUSION_METHODS, default="Linear Blending")
    p.add_argument("--regression-threshold", type=float, default=0.30)
    p.add_argument("--max-avg-displacement", type=float, default=2.50)
    p.add_argument("--absolute-displacement", type=float, default=3.50)
    p.add_argument("--no-compute-overlap", action="store_true")
    p.add_argument("--no-subpixel", action="store_true")
    p.add_argument("--computation", choices=("auto",) + tuple(COMPUTATION), default="auto")
    p.add_argument("--extra-options", default="", help="Extra plugin options, verbatim.")
    p.add_argument("--grayscale", action="store_true", help="Save the mosaic in grayscale.")
    p.add_argument("--downsample", type=int, default=1,
                   help="Reduce each tile by this integer factor before stitching.")
    p.add_argument("--tiles-to-gray", action="store_true",
                   help="Convert the tiles to 8-bit gray before stitching.")
    p.add_argument("--no-size-check", action="store_true",
                   help="Do not stop when the estimated mosaic is too large.")
    p.add_argument("--memory", default=None, help="Java heap for Fiji, e.g. 8g.")
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args()


def stitch_from_args(args):
    """Call ``stitch`` with the options of an ``argparse`` namespace."""
    return stitch(
        args.tiles_dir, output_path=args.output, mode=args.mode, pattern=args.pattern,
        grid_type=args.grid_type, order=args.order, grid_size=args.grid_size,
        tile_overlap=args.overlap, first_index=args.first_index,
        file_names=args.file_names, layout_file=args.layout_file,
        fusion_method=args.fusion, regression_threshold=args.regression_threshold,
        max_avg_displacement_threshold=args.max_avg_displacement,
        absolute_displacement_threshold=args.absolute_displacement,
        compute_overlap=not args.no_compute_overlap,
        subpixel_accuracy=not args.no_subpixel, computation=args.computation,
        extra_options=args.extra_options, grayscale=args.grayscale,
        downsample=args.downsample, tiles_to_gray=args.tiles_to_gray,
        check_size=not args.no_size_check,
        fiji_path=args.fiji, memory=args.memory, timeout=args.timeout,
        keep_temp=args.keep_temp, verbose=not args.quiet)


if __name__ == "__main__":
    stitch_from_args(_parse_args())

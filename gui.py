"""
gui
===

Graphical interface for pics2graph, built on napari.

    python gui.py                    # empty viewer
    python gui.py path/to/image.tif  # open a micrograph directly

The panel on the right has one tab per stage of the pipeline:

    1. Image      - open a micrograph / binary image, or stitch tiles with Fiji.
    2. Binarize   - adaptive binarization with live preview. The result is an
                    editable layer: paint to add wire pixels, erase to remove.
    3. Electrodes - paint (or load) the Vin / Vout electrode masks.
    4. Graph      - extract the graph, view the spatial graph G or the line
                    graph L over the image, and export it.

Every layer can be toggled, zoomed and edited with napari's own tools; the
panel only adds the pipeline steps on top.

Startup: only Qt is imported at first, so a loading window appears right away;
napari and the image-analysis libraries (which can take tens of seconds the
first time after installing or rebooting) are loaded afterwards, reporting each
step both in that window and in the terminal.
"""

import os
import sys
import time

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QFont, QPainter, QPixmap
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QSpinBox, QSplashScreen, QTabWidget, QVBoxLayout, QWidget,
)

_T0 = time.time()
_LOADED = False


def _load_modules(report=print):
    """
    Import the heavy dependencies (napari, OpenCV, scikit-image, skan,
    matplotlib) into this module's namespace, calling ``report(msg)`` before
    each step so the user can see that the program is loading, not frozen.
    """
    global _LOADED, cv2, np, napari, thread_worker, show_error, show_info, show_warning
    global pre, st, SkeletonAnalysis
    global get_line_graph_positions, get_node_positions
    global plot_line_graph_on_image, plot_network_on_image
    if _LOADED:
        return

    report("Loading napari (the slowest step, especially the first time)...")
    import napari
    from napari.qt.threading import thread_worker
    from napari.utils.notifications import show_error, show_info, show_warning

    report("Loading OpenCV and NumPy...")
    import cv2
    import numpy as np

    report("Loading scikit-image and skan...")
    import preprocessing as pre
    from SkeletonAnalysis import SkeletonAnalysis

    report("Loading matplotlib and the stitching module...")
    import matplotlib
    matplotlib.use("Agg")       # figures are only saved to disk, never shown
    import stitching as st
    from visualization import (get_line_graph_positions, get_node_positions,
                               plot_line_graph_on_image, plot_network_on_image)
    _LOADED = True

# Layer names
RAW = "micrograph"
BINARY = "binary"
ELECTRODES = "electrodes"
SKELETON = "skeleton"
EDGES = "graph edges"
NODES = "graph nodes"

VIN, VOUT = 1, 2            # label values in the electrodes layer

IMAGE_FILTER = "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.JPG *.TIF);;All files (*)"


# =============================================================================
#  I/O helpers (work with non-ASCII paths on Windows, unlike cv2.imread)
# =============================================================================
def imread(path, flags=-1):          # -1 == cv2.IMREAD_UNCHANGED
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


def imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".tif"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Could not write {path}")
    buf.tofile(path)


def to_rgb(img):
    """OpenCV image (gray / BGR / BGRA) -> gray or RGB for display."""
    if img.ndim == 3:
        img = img[..., :3][..., ::-1]
    return np.ascontiguousarray(img)


def to_gray_uint8(img):
    """Gray or RGB image of any dtype -> uint8 grayscale."""
    if img.ndim == 3:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _label_colormap(colors):
    """Colormap for a labels layer, compatible with old and new napari."""
    try:
        from napari.utils.colormaps import DirectLabelColormap
        return DirectLabelColormap(color_dict={None: "transparent", 0: "transparent", **colors})
    except ImportError:
        return None


# =============================================================================
#  Small widget helpers
# =============================================================================
def _spin(lo, hi, value, step=1):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(value)
    return w


def _dspin(lo, hi, value, step=0.05, decimals=2):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    return w


def _combo(items, current=None):
    w = QComboBox()
    w.addItems(list(items))
    # let the panel shrink instead of growing to the longest item
    w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    w.setMinimumContentsLength(8)
    if current is not None:
        w.setCurrentText(current)
    return w


def _scrollable(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area


# =============================================================================
#  Main panel
# =============================================================================
class Pics2GraphWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.raw_path = None        # path of the micrograph, if any
        self.raw_gray = None        # uint8 grayscale copy used for binarization
        self.SA = None              # last SkeletonAnalysis
        self._workers = []          # keep references to running threads

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self.binarize)

        self.tabs = QTabWidget()
        self.tabs.addTab(_scrollable(self._build_image_tab()), "1. Image")
        self.tabs.addTab(_scrollable(self._build_binarize_tab()), "2. Binarize")
        self.tabs.addTab(_scrollable(self._build_electrodes_tab()), "3. Electrodes")
        self.tabs.addTab(_scrollable(self._build_graph_tab()), "4. Graph")

        self.status = QLabel("Open a micrograph, a binary image, or stitch a folder of tiles.")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.status)
        self.setMinimumWidth(400)

    # -------------------------------------------------------------------------
    #  Generic helpers
    # -------------------------------------------------------------------------
    def _say(self, msg, level="info"):
        self.status.setText(msg)
        {"info": show_info, "warning": show_warning, "error": show_error}[level](msg)

    def _layer(self, name):
        return self.viewer.layers[name] if name in self.viewer.layers else None

    def _remove(self, *names):
        for n in names:
            if n in self.viewer.layers:
                self.viewer.layers.remove(n)

    def _shape(self):
        """(H, W) of the current image (binary or micrograph)."""
        for name in (BINARY, RAW):
            layer = self._layer(name)
            if layer is not None:
                return layer.data.shape[:2]
        return None

    def _run(self, fn, on_done, button, busy_msg):
        """Run ``fn`` in a background thread, then ``on_done(result)`` in the GUI."""
        worker = thread_worker(fn)()
        button.setEnabled(False)
        self.status.setText(busy_msg)

        def _errored(e):
            self._say(f"{type(e).__name__}: {e}", "error")

        def _finished():
            button.setEnabled(True)
            if worker in self._workers:
                self._workers.remove(worker)

        worker.returned.connect(on_done)
        worker.errored.connect(_errored)
        worker.finished.connect(_finished)
        self._workers.append(worker)
        worker.start()
        return worker

    def _path_row(self, kind="open", file_filter=IMAGE_FILTER, placeholder=""):
        """A line edit with a '...' browse button. ``kind``: open, save or dir."""
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setMinimumWidth(60)
        btn = QPushButton("...")
        btn.setFixedWidth(30)

        def browse():
            start = edit.text() or os.getcwd()
            if kind == "dir":
                path = QFileDialog.getExistingDirectory(self, "Choose folder", start)
            elif kind == "save":
                path, _ = QFileDialog.getSaveFileName(self, "Save as", start, file_filter)
            else:
                path, _ = QFileDialog.getOpenFileName(self, "Open", start, file_filter)
            if path:
                edit.setText(path)

        btn.clicked.connect(browse)
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(edit)
        lay.addWidget(btn)
        return row, edit

    # -------------------------------------------------------------------------
    #  Tab 1: Image / stitching
    # -------------------------------------------------------------------------
    def _build_image_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        box = QGroupBox("Open")
        b = QVBoxLayout(box)
        btn_raw = QPushButton("Open micrograph...")
        btn_raw.clicked.connect(self._ask_open_raw)
        btn_bin = QPushButton("Open binary image (skip binarization)...")
        btn_bin.clicked.connect(self._ask_open_binary)
        b.addWidget(btn_raw)
        b.addWidget(btn_bin)
        lay.addWidget(box)

        box = QGroupBox("Stitch tiles with Fiji")
        f = QFormLayout(box)
        row, self.st_tiles = self._path_row("dir", placeholder="folder with the tiles")
        f.addRow("Tiles folder", row)
        self.st_pattern = QLineEdit("*")
        f.addRow("File pattern", self.st_pattern)
        self.st_mode = _combo(["unknown", "grid", "positions"])
        self.st_mode.setToolTip("unknown: the plugin finds the arrangement by itself\n"
                                "grid: regular grid acquisition\n"
                                "positions: approximate positions from a TileConfiguration file")
        f.addRow("Mode", self.st_mode)

        self.st_grid_box = QGroupBox("Grid")
        g = QFormLayout(self.st_grid_box)
        self.st_grid_type = _combo(st.GRID_TYPES, "snake-by-rows")
        self.st_order = _combo(st.ROW_ORDERS)
        self.st_grid_type.currentTextChanged.connect(self._update_orders)
        self.st_cols = _spin(1, 1000, 2)
        self.st_rows = _spin(1, 1000, 2)
        self.st_overlap = _dspin(0, 99, 20, 1, 1)
        self.st_file_names = QLineEdit()
        self.st_file_names.setPlaceholderText("e.g. Image_{iiiii}.tif")
        self.st_first = _spin(0, 10 ** 9, 1)
        g.addRow("Type", self.st_grid_type)
        g.addRow("Order", self.st_order)
        g.addRow("Columns", self.st_cols)
        g.addRow("Rows", self.st_rows)
        g.addRow("Overlap (%)", self.st_overlap)
        g.addRow("File names", self.st_file_names)
        g.addRow("First index", self.st_first)
        f.addRow(self.st_grid_box)

        self.st_layout_box = QGroupBox("Positions")
        p = QFormLayout(self.st_layout_box)
        row, self.st_layout = self._path_row(
            "open", "Text files (*.txt);;All files (*)",
            placeholder="default: <tiles>/TileConfiguration.txt")
        p.addRow("Layout file", row)
        f.addRow(self.st_layout_box)

        self.st_fusion = _combo(st.FUSION_METHODS, "Linear Blending")
        self.st_regression = _dspin(0, 1, 0.30)
        self.st_maxavg = _dspin(0, 100, 2.50)
        self.st_absolute = _dspin(0, 100, 3.50)
        self.st_compute_overlap = QCheckBox("Compute overlap")
        self.st_compute_overlap.setChecked(True)
        self.st_subpixel = QCheckBox("Subpixel accuracy")
        self.st_subpixel.setChecked(True)
        self.st_computation = _combo(st.COMPUTATION, "time")
        self.st_extra = QLineEdit()
        self.st_extra.setPlaceholderText("extra plugin options (verbatim)")
        f.addRow("Fusion", self.st_fusion)
        f.addRow("Regression threshold", self.st_regression)
        f.addRow("Max/avg displacement", self.st_maxavg)
        f.addRow("Absolute displacement", self.st_absolute)
        f.addRow(self.st_compute_overlap)
        f.addRow(self.st_subpixel)
        f.addRow("Optimize for", self.st_computation)
        f.addRow("Extra options", self.st_extra)

        row, self.st_fiji = self._path_row("dir", placeholder="Fiji.app folder")
        try:
            self.st_fiji.setText(st.find_fiji())
        except FileNotFoundError:
            pass
        f.addRow("Fiji", row)
        self.st_memory = QLineEdit()
        self.st_memory.setPlaceholderText("e.g. 8g (default: Fiji's)")
        f.addRow("Java memory", self.st_memory)
        row, self.st_output = self._path_row(
            "save", "TIFF (*.tif)", placeholder="default: <tiles>/Fused.tif")
        f.addRow("Output", row)

        self.btn_stitch = QPushButton("Stitch")
        self.btn_stitch.clicked.connect(self.stitch)
        f.addRow(self.btn_stitch)
        lay.addWidget(box)
        lay.addStretch()

        self.st_mode.currentTextChanged.connect(self._update_stitch_mode)
        self._update_stitch_mode()
        return w

    def _update_orders(self):
        orders = st.COLUMN_ORDERS if "column" in self.st_grid_type.currentText() else st.ROW_ORDERS
        self.st_order.clear()
        self.st_order.addItems(list(orders))

    def _update_stitch_mode(self):
        mode = self.st_mode.currentText()
        self.st_grid_box.setVisible(mode == "grid")
        self.st_layout_box.setVisible(mode == "positions")
        self.st_compute_overlap.setEnabled(mode != "unknown")

    def _ask_open_raw(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open micrograph", self.raw_path or "", IMAGE_FILTER)
        if path:
            self.open_raw(path)

    def _ask_open_binary(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open binary image", "", IMAGE_FILTER)
        if path:
            self.open_binary(path)

    def open_raw(self, path):
        img = to_rgb(imread(path))
        self.raw_path = path
        self.raw_gray = to_gray_uint8(img)
        self._remove(RAW)
        self.viewer.add_image(img, name=RAW, rgb=img.ndim == 3)
        self.viewer.layers.move(self.viewer.layers.index(RAW), 0)   # keep it at the bottom
        self.viewer.reset_view()
        h, w = img.shape[:2]
        self._say(f"Opened {os.path.basename(path)} ({w} x {h} px). Next: 2. Binarize.")

    def open_binary(self, path):
        mask = imread(path, cv2.IMREAD_GRAYSCALE) > 0
        self.set_binary(mask)
        if self._layer(RAW) is None:
            self.viewer.reset_view()
        self._say(f"Opened binary image {os.path.basename(path)}. Next: 3. Electrodes or 4. Graph.")

    def stitch(self):
        tiles = self.st_tiles.text().strip()
        if not tiles or not os.path.isdir(tiles):
            self._say("Choose the folder with the tiles.", "warning")
            return
        mode = self.st_mode.currentText()
        kwargs = dict(
            output_path=self.st_output.text().strip() or None,
            mode=mode, pattern=self.st_pattern.text().strip() or "*",
            fusion_method=self.st_fusion.currentText(),
            regression_threshold=self.st_regression.value(),
            max_avg_displacement_threshold=self.st_maxavg.value(),
            absolute_displacement_threshold=self.st_absolute.value(),
            compute_overlap=self.st_compute_overlap.isChecked(),
            subpixel_accuracy=self.st_subpixel.isChecked(),
            computation=self.st_computation.currentText(),
            extra_options=self.st_extra.text().strip(),
            fiji_path=self.st_fiji.text().strip() or None,
            memory=self.st_memory.text().strip() or None,
            verbose=True,
        )
        if mode == "grid":
            kwargs.update(grid_type=self.st_grid_type.currentText(),
                          order=self.st_order.currentText(),
                          grid_size=(self.st_cols.value(), self.st_rows.value()),
                          tile_overlap=self.st_overlap.value(),
                          file_names=self.st_file_names.text().strip() or None,
                          first_index=self.st_first.value())
        elif mode == "positions":
            kwargs["layout_file"] = self.st_layout.text().strip() or None

        def job():
            return st.stitch(tiles, **kwargs)

        self._run(job, self._stitch_done, self.btn_stitch,
                  "Stitching with Fiji... (progress is printed in the terminal)")

    def _stitch_done(self, path):
        self.open_raw(path)
        self._say(f"Mosaic saved to {path}. Next: 2. Binarize.")

    # -------------------------------------------------------------------------
    #  Tab 2: Binarize
    # -------------------------------------------------------------------------
    def _build_binarize_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        box = QGroupBox("Adaptive binarization")
        f = QFormLayout(box)
        self.bin_block = _spin(3, 4001, 151, 2)
        self.bin_block.setToolTip("Size of the local window (odd).")
        self.bin_C = _spin(-255, 255, 15)
        self.bin_C.setToolTip("Constant subtracted from the local mean: higher = fewer wire pixels.")
        self.bin_thresh = _spin(0, 255, 55)
        self.bin_thresh.setToolTip("Pixels darker than this are always background.")
        self.bin_live = QCheckBox("Live preview")
        self.bin_live.setChecked(True)
        f.addRow("Block size", self.bin_block)
        f.addRow("C", self.bin_C)
        f.addRow("Intensity threshold", self.bin_thresh)
        f.addRow(self.bin_live)
        btn = QPushButton("Binarize")
        btn.clicked.connect(self.binarize)
        f.addRow(btn)
        for s in (self.bin_block, self.bin_C, self.bin_thresh):
            s.valueChanged.connect(self._schedule_preview)
        lay.addWidget(box)

        box = QGroupBox("Edit / save")
        v = QVBoxLayout(box)
        help_ = QLabel(
            "The 'binary' layer is editable: select it and use napari's brush "
            "(label 1) to add wire pixels or the eraser to remove them. "
            "Lower its opacity or toggle it to compare with the micrograph.")
        help_.setWordWrap(True)
        v.addWidget(help_)
        btn = QPushButton("Save binary image...")
        btn.clicked.connect(self.save_binary)
        v.addWidget(btn)
        lay.addWidget(box)
        lay.addStretch()
        return w

    def _schedule_preview(self):
        if self.bin_live.isChecked() and self.raw_gray is not None:
            self._preview_timer.start()

    def binarize(self):
        if self.raw_gray is None:
            self._say("Open a micrograph first (tab 1).", "warning")
            return
        block = self.bin_block.value()
        if block % 2 == 0:
            block += 1
            self.bin_block.blockSignals(True)
            self.bin_block.setValue(block)
            self.bin_block.blockSignals(False)
        mask = pre.adaptive_binarize(self.raw_gray.copy(), block_size=block,
                                     C=self.bin_C.value(),
                                     intensity_thresh=self.bin_thresh.value())
        self.set_binary(mask > 0)
        frac = 100.0 * np.count_nonzero(mask) / mask.size
        self.status.setText(f"Binarized: {frac:.1f} % of pixels are wire. "
                            "Adjust the parameters or go to 3. Electrodes / 4. Graph.")

    def set_binary(self, mask):
        data = np.asarray(mask, dtype=np.uint8)
        layer = self._layer(BINARY)
        if layer is not None and layer.data.shape == data.shape:
            layer.data = data
            return
        self._remove(BINARY)
        layer = self.viewer.add_labels(data, name=BINARY, opacity=0.6)
        cmap = _label_colormap({1: "yellow"})
        if cmap is not None:
            layer.colormap = cmap
        layer.selected_label = 1

    def binary_mask(self):
        layer = self._layer(BINARY)
        return None if layer is None else (np.asarray(layer.data) > 0)

    def save_binary(self):
        mask = self.binary_mask()
        if mask is None:
            self._say("There is no binary image yet.", "warning")
            return
        default = (os.path.splitext(self.raw_path)[0] + "_binary.tif") if self.raw_path else ""
        path, _ = QFileDialog.getSaveFileName(self, "Save binary image", default, "TIFF (*.tif)")
        if path:
            imwrite(path, mask.astype(np.uint8) * 255)
            self._say(f"Binary image saved to {path}.")

    # -------------------------------------------------------------------------
    #  Tab 3: Electrodes
    # -------------------------------------------------------------------------
    def _build_electrodes_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        box = QGroupBox("Draw")
        v = QVBoxLayout(box)
        help_ = QLabel(
            "Paint the input electrode (Vin, red) and the output electrode "
            "(Vout, blue) over the image. Use the polygon tool for large areas, "
            "or fill to flood a closed contour.")
        help_.setWordWrap(True)
        v.addWidget(help_)
        btn = QPushButton("Create / clear electrode layer")
        btn.clicked.connect(self.new_electrodes)
        v.addWidget(btn)

        f = QFormLayout()
        self.el_label = _combo(["Vin (red)", "Vout (blue)"])
        self.el_label.currentIndexChanged.connect(lambda _: self._set_el_tool(None))
        f.addRow("Electrode", self.el_label)
        self.el_brush = _spin(1, 500, 30)
        self.el_brush.valueChanged.connect(self._set_brush)
        f.addRow("Brush size", self.el_brush)
        v.addLayout(f)
        row = QHBoxLayout()
        for name, mode in (("Brush", "paint"), ("Polygon", "polygon"),
                           ("Fill", "fill"), ("Erase", "erase")):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, m=mode: self._set_el_tool(m))
            row.addWidget(b)
        v.addLayout(row)
        lay.addWidget(box)

        box = QGroupBox("Load / save masks")
        f = QFormLayout(box)
        row, self.el_vin = self._path_row("open", placeholder="Vin mask")
        f.addRow("Vin", row)
        row, self.el_vout = self._path_row("open", placeholder="Vout mask")
        f.addRow("Vout", row)
        btn = QPushButton("Load masks")
        btn.clicked.connect(self.load_electrodes)
        f.addRow(btn)
        btn = QPushButton("Save masks...")
        btn.clicked.connect(self.save_electrodes)
        f.addRow(btn)
        lay.addWidget(box)
        lay.addStretch()
        return w

    def _electrodes_layer(self, create=True):
        layer = self._layer(ELECTRODES)
        if layer is None and create:
            shape = self._shape()
            if shape is None:
                self._say("Open an image first (tab 1).", "warning")
                return None
            layer = self.viewer.add_labels(np.zeros(shape, np.uint8), name=ELECTRODES, opacity=0.4)
            cmap = _label_colormap({VIN: "red", VOUT: "blue"})
            if cmap is not None:
                layer.colormap = cmap
            layer.brush_size = self.el_brush.value()
        return layer

    def new_electrodes(self):
        layer = self._layer(ELECTRODES)
        if layer is not None:
            layer.data = np.zeros_like(layer.data)
        else:
            layer = self._electrodes_layer()
        if layer is not None:
            self._set_el_tool("paint")

    def _set_el_tool(self, mode):
        layer = self._electrodes_layer()
        if layer is None:
            return
        self.viewer.layers.selection.active = layer
        layer.selected_label = VIN if self.el_label.currentIndex() == 0 else VOUT
        if mode is not None:
            layer.mode = mode

    def _set_brush(self, size):
        layer = self._layer(ELECTRODES)
        if layer is not None:
            layer.brush_size = size

    def load_electrodes(self):
        p1, p2 = self.el_vin.text().strip(), self.el_vout.text().strip()
        if not (p1 and p2):
            self._say("Choose both mask files.", "warning")
            return
        m1 = imread(p1, cv2.IMREAD_GRAYSCALE) > 0
        m2 = imread(p2, cv2.IMREAD_GRAYSCALE) > 0
        shape = self._shape()
        if shape is not None and m1.shape != tuple(shape):
            self._say(f"The masks {m1.shape} do not match the image {tuple(shape)}.", "error")
            return
        layer = self._electrodes_layer()
        if layer is None:
            return
        data = np.zeros(m1.shape, np.uint8)
        data[m1] = VIN
        data[m2] = VOUT
        layer.data = data
        self._say("Electrode masks loaded.")

    def electrode_masks(self):
        """(Vin, Vout) masks as uint8 0/255, or None if nothing was painted."""
        layer = self._layer(ELECTRODES)
        if layer is None:
            return None
        data = np.asarray(layer.data)
        m1, m2 = data == VIN, data == VOUT
        if not m1.any() or not m2.any():
            return None
        return m1.astype(np.uint8) * 255, m2.astype(np.uint8) * 255

    def save_electrodes(self):
        masks = self.electrode_masks()
        if masks is None:
            self._say("Paint both electrodes first.", "warning")
            return
        default = os.path.splitext(self.raw_path)[0] if self.raw_path else ""
        base, _ = QFileDialog.getSaveFileName(
            self, "Base name for the masks (adds _electrode_mask1/2.tif)", default, "TIFF (*.tif)")
        if base:
            base = os.path.splitext(base)[0]
            imwrite(base + "_electrode_mask1.tif", masks[0])
            imwrite(base + "_electrode_mask2.tif", masks[1])
            self._say(f"Saved {base}_electrode_mask1.tif and _electrode_mask2.tif.")

    # -------------------------------------------------------------------------
    #  Tab 4: Graph
    # -------------------------------------------------------------------------
    def _build_graph_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        box = QGroupBox("Extract")
        f = QFormLayout(box)
        self.g_clean = QCheckBox("Clean network (prune degree 1, collapse degree 2)")
        self.g_clean.setChecked(True)
        self.g_use_el = QCheckBox("Use electrodes")
        self.g_use_el.setChecked(True)
        f.addRow(self.g_clean)
        f.addRow(self.g_use_el)
        self.btn_extract = QPushButton("Extract graph")
        self.btn_extract.clicked.connect(self.extract_graph)
        f.addRow(self.btn_extract)
        self.g_stats = QLabel("")
        self.g_stats.setWordWrap(True)
        f.addRow(self.g_stats)
        lay.addWidget(box)

        box = QGroupBox("View")
        f = QFormLayout(box)
        self.g_view = _combo(["Spatial graph G (nodes = junctions)",
                              "Line graph L (nodes = wires)"])
        self.g_view.currentIndexChanged.connect(lambda _: self.draw_graph())
        self.g_node_size = _spin(1, 200, 8)
        self.g_edge_width = _dspin(0.5, 50, 2.0, 0.5, 1)
        self.g_node_size.valueChanged.connect(lambda _: self.draw_graph())
        self.g_edge_width.valueChanged.connect(lambda _: self.draw_graph())
        self.g_skeleton = QCheckBox("Show skeleton")
        self.g_skeleton.toggled.connect(self._toggle_skeleton)
        f.addRow("Show", self.g_view)
        f.addRow("Node size", self.g_node_size)
        f.addRow("Edge width", self.g_edge_width)
        f.addRow(self.g_skeleton)
        lay.addWidget(box)

        box = QGroupBox("Export")
        f = QFormLayout(box)
        self.g_save_kind = _combo(["line graph L", "spatial graph G"])
        f.addRow("Graph to save", self.g_save_kind)
        btn = QPushButton("Save graph (.graphml)...")
        btn.clicked.connect(self.save_graph)
        f.addRow(btn)
        btn = QPushButton("Save figure (graph over image)...")
        btn.clicked.connect(self.save_figure)
        f.addRow(btn)
        btn = QPushButton("Save screenshot of the view...")
        btn.clicked.connect(self.save_screenshot)
        f.addRow(btn)
        lay.addWidget(box)
        lay.addStretch()
        return w

    def extract_graph(self):
        mask = self.binary_mask()
        if mask is None:
            self._say("There is no binary image yet (tabs 1-2).", "warning")
            return
        binary = mask.astype(np.uint8) * 255
        masks = self.electrode_masks() if self.g_use_el.isChecked() else None
        if self.g_use_el.isChecked() and masks is None:
            show_warning("No electrodes painted (both Vin and Vout are needed): "
                         "extracting the graph without electrodes.")
        clean = self.g_clean.isChecked()
        raw_path = self.raw_path

        def job():
            SA = SkeletonAnalysis(img=binary)
            SA.original_img_path = raw_path
            SA.img_cleaning()
            if masks is not None:
                SA.load_electrodes(*masks)
            SA.skeletonize()
            SA.get_network()
            if clean:
                SA.clean_network()
            SA.convert_to_line_graph()
            return SA

        self._run(job, self._graph_done, self.btn_extract, "Extracting graph...")

    def _graph_done(self, SA):
        self.SA = SA
        G, L = SA.G, SA.L
        n_vin = sum(1 for n in G if isinstance(n, str) and n.startswith("Vin_"))
        n_vout = sum(1 for n in G if isinstance(n, str) and n.startswith("Vout_"))
        txt = (f"G: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n"
               f"L: {L.number_of_nodes()} nodes, {L.number_of_edges()} edges")
        if n_vin or n_vout:
            txt += f"\nElectrode contacts: {n_vin} Vin, {n_vout} Vout"
        self.g_stats.setText(txt)
        if self.g_skeleton.isChecked():
            self._toggle_skeleton(True)
        self.draw_graph()
        self._say("Graph extracted. Next: export it below.")

    def draw_graph(self):
        SA = self.SA
        if SA is None:
            return
        line = self.g_view.currentIndex() == 1
        if line:
            graph, pos = SA.L, get_line_graph_positions(SA)

            def is_el(n, p):
                return any(isinstance(m, str) and m.startswith(p) for m in (n[0], n[1]))
        else:
            graph, pos = SA.G, get_node_positions(SA)

            def is_el(n, p):
                return isinstance(n, str) and n.startswith(p)

        nodes = list(graph.nodes)
        index = {n: i for i, n in enumerate(nodes)}
        coords = np.array([[pos[n][1], pos[n][0]] for n in nodes], dtype=float).reshape(-1, 2)
        colors, sizes = [], []
        base = self.g_node_size.value()
        for n in nodes:
            if is_el(n, "Vin_"):
                colors.append("red"); sizes.append(base * 2)
            elif is_el(n, "Vout_"):
                colors.append("blue"); sizes.append(base * 2)
            else:
                colors.append("deepskyblue"); sizes.append(base)

        edges = np.array([[index[u], index[v]] for u, v in graph.edges], dtype=int).reshape(-1, 2)
        vectors = np.zeros((len(edges), 2, 2))
        if len(edges):
            vectors[:, 0] = coords[edges[:, 0]]
            vectors[:, 1] = coords[edges[:, 1]] - coords[edges[:, 0]]

        self._remove(EDGES, NODES)
        self.viewer.add_vectors(vectors, name=EDGES, edge_width=self.g_edge_width.value(),
                                edge_color="magenta" if line else "white")
        try:
            self.viewer.add_points(coords, name=NODES, size=sizes, face_color=colors,
                                   border_width=0)
        except TypeError:                                   # older napari
            self.viewer.add_points(coords, name=NODES, size=sizes, face_color=colors,
                                   edge_width=0)
        # fade the layers underneath so the graph stands out
        for name, opacity in ((BINARY, 0.3), (ELECTRODES, 0.25)):
            layer = self._layer(name)
            if layer is not None and layer.opacity > opacity:
                layer.opacity = opacity
        self.viewer.layers.selection.active = self._layer(NODES)

    def _toggle_skeleton(self, show):
        if not show:
            self._remove(SKELETON)
            return
        if self.SA is None or getattr(self.SA, "skeleton_img", None) is None:
            return
        self._remove(SKELETON)
        self.viewer.add_image(self.SA.skeleton_img.astype(np.uint8), name=SKELETON,
                              colormap="gray", blending="additive", contrast_limits=(0, 1))
        for name in (EDGES, NODES):          # skeleton under the graph
            if name in self.viewer.layers:
                self.viewer.layers.move(self.viewer.layers.index(name), len(self.viewer.layers))

    def save_graph(self):
        if self.SA is None:
            self._say("Extract the graph first.", "warning")
            return
        line = self.g_save_kind.currentIndex() == 0
        default = (os.path.splitext(self.raw_path)[0] + ".graphml") if self.raw_path else "graph.graphml"
        path, _ = QFileDialog.getSaveFileName(self, "Save graph", default, "GraphML (*.graphml)")
        if path:
            self.SA.save_graph(path, save_line_graph=line)
            self._say(f"{'Line' if line else 'Spatial'} graph saved to {path}.")

    def save_figure(self):
        if self.SA is None:
            self._say("Extract the graph first.", "warning")
            return
        default = (os.path.splitext(self.raw_path)[0] + "_graph.png") if self.raw_path else "graph.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", default,
                                              "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        import matplotlib.pyplot as plt
        if self.g_view.currentIndex() == 1:
            ax = plot_line_graph_on_image(self.SA, save_path=path, dpi=400)
        else:
            ax = plot_network_on_image(self.SA, save_path=path, dpi=400)
        plt.close(ax.figure)
        self._say(f"Figure saved to {path}.")

    def save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "view.png", "PNG (*.png)")
        if path:
            self.viewer.screenshot(path, canvas_only=True)
            self._say(f"Screenshot saved to {path}.")


# =============================================================================
#  Entry point
# =============================================================================
class _Splash(QSplashScreen):
    """Small loading window shown while the heavy libraries are imported."""

    def __init__(self):
        pix = QPixmap(520, 170)
        pix.fill(QColor("#262930"))                      # napari's dark background
        p = QPainter(pix)
        p.setPen(QColor("#f0f1f2"))
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        p.setFont(font)
        p.drawText(28, 62, "pics2graph")
        font.setPointSize(10)
        font.setBold(False)
        p.setFont(font)
        p.setPen(QColor("#9da3ae"))
        p.drawText(30, 90, "photomicrograph -> graph")
        p.end()
        super().__init__(pix)
        self._start = time.time()

    def report(self, msg):
        elapsed = time.time() - self._start
        self.showMessage(f"   {msg}   ({elapsed:.0f} s)\n",
                         Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                         QColor("#f0f1f2"))
        QApplication.processEvents()


def _console(msg):
    print(f"[pics2graph {time.time() - _T0:5.1f} s] {msg}", flush=True)


def launch(image_path=None, show=True, report=None):
    """Create the viewer with the pics2graph panel and return (viewer, panel)."""
    report = report or _console
    _load_modules(report)
    report("Opening the window...")
    viewer = napari.Viewer(title="pics2graph", show=show)
    panel = Pics2GraphWidget(viewer)
    viewer.window.add_dock_widget(panel, name="pics2graph", area="right")
    if image_path:
        report(f"Opening {os.path.basename(image_path)}...")
        panel.open_raw(image_path)
    return viewer, panel


def main():
    _console("Starting (the first run after installing or rebooting can take "
             "up to a minute)...")
    app = QApplication.instance() or QApplication(sys.argv)
    splash = _Splash()
    splash.show()
    app.processEvents()

    def report(msg):
        _console(msg)
        splash.report(msg)

    try:
        viewer, _ = launch(sys.argv[1] if len(sys.argv) > 1 else None, report=report)
    except Exception:
        splash.close()
        raise
    splash.finish(viewer.window._qt_window)
    _console("Ready.")
    napari.run()


if __name__ == "__main__":
    main()

"""
SkeletonAnalysis
================

Extraction of the connection graph of a nanowire network from its binarized
photomicrography: morphological cleaning, skeletonization, graph extraction
(``skan`` -> networkx), pruning, line graph and export to .graphml.
"""

import cv2
import numpy as np
from skimage.morphology import opening, disk, skeletonize, dilation
from skimage.morphology import remove_small_holes as _remove_small_holes
from skimage.morphology import remove_small_objects as _remove_small_objects
from scipy.ndimage import binary_fill_holes
from skan import Skeleton
from skan.csr import skeleton_to_nx
import networkx as nx
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# scikit-image compatibility: since 0.26 these functions take ``max_size``
# (regions of size <= max_size are removed). Earlier versions took
# ``min_size`` / ``area_threshold`` (regions of size < threshold are removed),
# so the equivalent value is ``max_size + 1``.
# -----------------------------------------------------------------------------
def remove_small_holes(ar, max_size):
    try:
        return _remove_small_holes(ar, max_size=max_size)
    except TypeError:
        return _remove_small_holes(ar, area_threshold=max_size + 1)


def remove_small_objects(ar, max_size):
    try:
        return _remove_small_objects(ar, max_size=max_size)
    except TypeError:
        return _remove_small_objects(ar, min_size=max_size + 1)


def _as_gray_uint8(img):
    """Convert an array (bool, gray, RGB/RGBA, any dtype) to a uint8 grayscale image."""
    img = np.asarray(img)
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    if img.ndim == 3:
        img = img[..., :3]
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _load_mask(mask):
    """A mask given as a path or as an array, returned as a uint8 grayscale image."""
    if isinstance(mask, (str, bytes)) or hasattr(mask, "__fspath__"):
        return cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    return _as_gray_uint8(mask)


class SkeletonAnalysis:
    def __init__(self, img_path=None, img=None):
        """
        Load the binarized image, either from ``img_path`` or directly as an
        array with ``img`` (e.g. ``SkeletonAnalysis(img=binary_mask)``).
        """
        if img_path is None and img is None:
            raise ValueError("Pass img_path or img.")
        self.img_path = img_path
        if img is not None:
            self.img = _as_gray_uint8(img)
        else:
            self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.img_original = self.img.copy() if self.img is not None else None  # before img_cleaning, for plots
        self.original_img_path = None         # optional: path to the original micrograph
        self.has_electrodes = False

    def load_electrodes(self, path1, path2):
        """Load the Vin and Vout masks, given as paths or as arrays."""
        self.electrode_mask1 = _load_mask(path1)
        self.electrode_mask2 = _load_mask(path2)

        self.electrode_mask1 = binary_fill_holes(self.electrode_mask1)
        self.electrode_mask2 = binary_fill_holes(self.electrode_mask2)

        self.electrode_mask1 = remove_small_objects(self.electrode_mask1, max_size=500)
        self.electrode_mask2 = remove_small_objects(self.electrode_mask2, max_size=500)

        self.has_electrodes = True

    def img_cleaning(self):
        self.img = remove_small_holes(self.img.astype(bool), max_size=2)
        self.img = opening(self.img, footprint=disk(1))

    def skeletonize(self):
        self.skeleton_img = skeletonize(self.img)
        if self.has_electrodes:
            # Only remove the electrodes, without adding contact points to the skeleton
            self.skeleton_img = self.skeleton_img & ~self.electrode_mask1 & ~self.electrode_mask2

            # Keep the dilated masks to use them in get_network
            self.electrode_dil1 = dilation(self.electrode_mask1, disk(5))
            self.electrode_dil2 = dilation(self.electrode_mask2, disk(5))

    def get_network(self):
        skeleton = Skeleton(self.skeleton_img)
        self.G = nx.Graph(skeleton_to_nx(skeleton))
        self.node_coords = np.array(skeleton.coordinates)

        if self.has_electrodes:
            electrode_nodes1 = []
            electrode_nodes2 = []

            for node_id, (y, x) in enumerate(self.node_coords):
                yi, xi = int(y), int(x)
                if self.electrode_dil1[yi, xi] and not self.electrode_mask1[yi, xi] and not self.electrode_mask2[yi, xi]:
                    electrode_nodes1.append(node_id)
                if self.electrode_dil2[yi, xi] and not self.electrode_mask2[yi, xi] and not self.electrode_mask1[yi, xi]:
                    electrode_nodes2.append(node_id)

            self.electrode_pos = {}

            for i, n in enumerate(electrode_nodes1):
                node_name = f"Vin_{i}"
                y, x = self.node_coords[n]
                self.G.add_node(node_name)
                self.G.add_edge(node_name, n)
                self.electrode_pos[node_name] = (x, y)

            for i, n in enumerate(electrode_nodes2):
                node_name = f"Vout_{i}"
                y, x = self.node_coords[n]
                self.G.add_node(node_name)
                self.G.add_edge(node_name, n)
                self.electrode_pos[node_name] = (x, y)

        self.G.remove_edges_from(list(nx.selfloop_edges(self.G)))

    def clean_network(self):
        electrode_nodes = {n for n in self.G.nodes if isinstance(n, str) and
                           (n.startswith("Vin_") or n.startswith("Vout_"))}
        changed = True
        while changed:
            changed = False

            isolated = [n for n, d in self.G.degree() if d == 0]
            if isolated:
                self.G.remove_nodes_from(isolated)
                changed = True

            deg1_nodes = [n for n, d in self.G.degree() if d == 1 and n not in electrode_nodes]
            if deg1_nodes:
                self.G.remove_nodes_from(deg1_nodes)
                changed = True

            deg2_nodes = [n for n, d in self.G.degree() if d == 2 and n not in electrode_nodes]
            for n in deg2_nodes:
                neighbors = list(self.G.neighbors(n))
                if len(neighbors) == 2:
                    u, v = neighbors
                    self.G.add_edge(u, v)
                self.G.remove_node(n)
                changed = True

            edges_to_remove = [(u, v) for u, v in self.G.edges()
                               if isinstance(u, str) and isinstance(v, str)]
            if edges_to_remove:
                self.G.remove_edges_from(edges_to_remove)
                changed = True

    def convert_to_line_graph(self):
        self.L = nx.line_graph(self.G)
        self.L = nx.Graph(self.L)

    def save_graph(self, path, save_line_graph=True):
        if save_line_graph:
            nx.write_graphml(self.L, path)
        else:
            nx.write_graphml(self.G, path)

    def plot_graph(self, save_path=None):
        pos = {}

        for n in self.G.nodes:
            if isinstance(n, str) and (n.startswith("Vin_") or n.startswith("Vout_")):
                pos[n] = self.electrode_pos[n]
            else:
                y, x = self.node_coords[n]
                pos[n] = (x, y)

        plt.figure(figsize=(8, 8))
        plt.imshow(self.img, cmap='gray')

        vin_nodes = [n for n in self.G.nodes if isinstance(n, str) and n.startswith("Vin_")]
        vout_nodes = [n for n in self.G.nodes if isinstance(n, str) and n.startswith("Vout_")]
        network_nodes = [n for n in self.G.nodes if not (isinstance(n, str) and (n.startswith("Vin_") or n.startswith("Vout_")))]

        nx.draw_networkx_edges(self.G, pos, width=1.5, edge_color='green', node_size=25)
        nx.draw_networkx_nodes(self.G, pos, nodelist=network_nodes, node_size=25)
        nx.draw_networkx_nodes(self.G, pos, nodelist=vin_nodes, node_size=150, node_color='red')
        nx.draw_networkx_nodes(self.G, pos, nodelist=vout_nodes, node_size=150, node_color='blue')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.show()

    def complete_analysis(self, clean_network=True, electrodes_path=None, save_graph_path=None, plot=False, save_plot_path=None, verbose=False):
        if verbose:
            print("Starting complete analysis...")

        self.img_cleaning()
        if verbose:
            print("Image cleaned.")

        if electrodes_path:
            self.load_electrodes(*electrodes_path)
            if verbose:
                print("Electrode masks loaded.")

        self.skeletonize()
        if verbose:
            print("Skeleton generated.")

        self.get_network()
        if verbose:
            print("Network extracted from skeleton.")

        if clean_network:
            self.clean_network()
            if verbose:
                print("Network cleaned.")

        self.convert_to_line_graph()
        if verbose:
            print("Converted to line graph.")

        if save_graph_path:
            self.save_graph(save_graph_path)
            if verbose:
                print(f"Graph saved to {save_graph_path}.")

        if plot:
            self.plot_graph(save_plot_path)
            if verbose:
                print("Plot generated.")

        return

"""
visualization
=============

Plots of the graph extracted from an image, overlaid on the image itself.
"""

import os

import cv2
import networkx as nx
import matplotlib.pyplot as plt


def _is_electrode(n, prefix=("Vin_", "Vout_")):
    """True if node ``n`` is an electrode node (``Vin_*`` / ``Vout_*``)."""
    return isinstance(n, str) and n.startswith(prefix)


def _get_background(SA, background):
    """
    Return the background image to use in the plots.

    ``background`` can be:
        "skeleton" : the skeletonized image (``SA.skeleton_img``); if it does
                     not exist yet, falls back to the binary image.
        "img"      : the already cleaned binary image (``SA.img``).
        "raw"      : the image as loaded in ``SkeletonAnalysis``, before
                     ``img_cleaning`` (``SA.img_original``).
        "original" : the original grayscale micrograph. Taken from
                     ``SA.original_img_path`` if set; otherwise it is searched
                     next to the binary image by stripping the "_binary" suffix.
        str        : any other string is interpreted as a path to an image,
                     loaded in grayscale.
        None       : no background.
        ndarray    : a custom image.

    If the background does not have the same size as ``SA.img`` a warning is
    printed, because in that case the node coordinates will not line up.
    """
    if background is None:
        return None

    if isinstance(background, str):
        if background == "skeleton":
            bg = getattr(SA, "skeleton_img", SA.img)
        elif background == "img":
            bg = SA.img
        elif background == "raw":
            bg = getattr(SA, "img_original", SA.img)
        else:
            if background == "original":
                path = getattr(SA, "original_img_path", None)
                if path is None:
                    path = _guess_original_path(SA)
                if path is None:
                    raise ValueError(
                        "Original image not found. Pass the path directly "
                        "(background='.../M4.tif') or set SA.original_img_path."
                    )
            else:
                path = background
            bg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if bg is None:
                raise FileNotFoundError(f"Could not read the background image: {path}")
    else:
        bg = background

    ref = getattr(SA, "img", None)
    if ref is not None and hasattr(bg, "shape") and bg.shape[:2] != ref.shape[:2]:
        print(f"[warning] the background {bg.shape[:2]} does not match the "
              f"processed image {ref.shape[:2]}: the graph will not be aligned.")
    return bg


def _guess_original_path(SA):
    """
    Try to find the original micrograph from the path of the binary image:
    ``.../M4_binary.tif`` -> tries ``.../M4.tif``, ``.../M4.tiff``, ``.../M4.png``,
    ``.../M4.jpg``, ``.../M4.JPG``. Returns None if nothing is found.
    """
    binary_path = getattr(SA, "img_path", None)
    if not binary_path:
        return None
    base, _ = os.path.splitext(binary_path)
    for suffix in ("_binary", "_bin"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for ext in (".tif", ".tiff", ".png", ".jpg", ".JPG", ".jpeg"):
        cand = base + ext
        if os.path.exists(cand):
            return cand
    return None


def _zoom_factor(SA, xlim, ylim, zoom_scale="auto", clip=(1.0, 8.0)):
    """
    Magnification factor of the zoom relative to the full image, used to
    enlarge nodes and edges when looking at a small region.

    ``zoom_scale`` can be "auto" (computed from xlim/ylim), None or 1 (no
    scaling) or a number (fixed factor, chosen by hand).

    With ``aspect="equal"`` the effective magnification is set by the least
    magnified axis, so the minimum of the two ratios is used. It is clipped to
    ``clip`` so that a very small zoom does not produce giant nodes.
    """
    if zoom_scale is None:
        return 1.0
    if zoom_scale != "auto":
        return float(zoom_scale)

    height, width = SA.img.shape[:2]
    ratios = []
    if xlim:
        ratios.append(width / abs(xlim[1] - xlim[0]))
    if ylim:
        ratios.append(height / abs(ylim[1] - ylim[0]))
    if not ratios:
        return 1.0
    f = min(ratios)
    return max(clip[0], min(f, clip[1]))


def plot_network_on_image(SA, xlim=None, ylim=None, transparent=False,
                          save_path=None, dpi=200, ax=None,
                          background="skeleton", bg_cmap="gray", bg_alpha=1.0,
                          node_size=25, edge_width=1.5, zoom_scale="auto"):
    """
    Draw the graph of a ``SkeletonAnalysis`` on top of its image, with the
    Vin (red) and Vout (blue) electrodes highlighted.

    Parameters
    ----------
    SA : SkeletonAnalysis
        Already processed instance (with ``G``, ``node_coords``, ``electrode_pos``, ``img``).
    xlim, ylim : tuple, optional
        Limits to zoom into a region.
    transparent : bool
        If True, hides the background image and makes the figure transparent
        (useful to export only the graph).
    save_path : str, optional
        If given, saves the figure.
    dpi : int
        Resolution when saving.
    ax : matplotlib Axes, optional
        Axes to draw on; if not given, a new one is created.
    background : {"skeleton", "img", "raw", "original"}, path, None or ndarray
        Background image. Defaults to the skeleton (``SA.skeleton_img``).
        See ``_get_background``.
    bg_cmap : str
        Background colormap ("gray", "gray_r", "bone", ...).
    bg_alpha : float
        Background opacity; lowering it helps the graph stand out over the
        original micrograph.
    node_size, edge_width :
        Base size of nodes and edges at zoom 1 (full image).
    zoom_scale : "auto", None or float
        Scaling of nodes/edges with zoom. "auto" enlarges them according to how
        much xlim/ylim shrink the visible region (nodes ~ f, edges ~ sqrt(f),
        with f clipped at 8x); None keeps them fixed; a number forces the factor.
    """
    pos = get_node_positions(SA)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    bg = _get_background(SA, background)
    if bg is not None:
        ax.imshow(bg, cmap=bg_cmap, alpha=0 if transparent else bg_alpha)

    vin_nodes = [n for n in SA.G.nodes if _is_electrode(n, "Vin_")]
    vout_nodes = [n for n in SA.G.nodes if _is_electrode(n, "Vout_")]
    network_nodes = [n for n in SA.G.nodes if not _is_electrode(n)]

    f = _zoom_factor(SA, xlim, ylim, zoom_scale)
    ns, ne, nw = node_size * f, 150 * f, edge_width * f ** 0.5

    nx.draw_networkx_edges(SA.G, pos, ax=ax, width=nw, edge_color="green")
    nx.draw_networkx_nodes(SA.G, pos, ax=ax, nodelist=network_nodes, node_size=ns, node_color="C0")
    nx.draw_networkx_nodes(SA.G, pos, ax=ax, nodelist=vin_nodes, node_size=ne, node_color="red")
    nx.draw_networkx_nodes(SA.G, pos, ax=ax, nodelist=vout_nodes, node_size=ne, node_color="blue")

    ax.axis("off")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if transparent:
        ax.figure.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    return ax


def get_node_positions(SA):
    """
    Dictionary ``{node: (x, y)}`` with the position of each node of ``SA.G``.

    Network nodes come from ``SA.node_coords`` (stored as (y, x)) and the
    Vin_/Vout_ electrode nodes from ``SA.electrode_pos`` (already (x, y)).
    """
    pos = {}
    for n in SA.G.nodes:
        if _is_electrode(n):
            pos[n] = SA.electrode_pos[n]
        else:
            y, x = SA.node_coords[n]
            pos[n] = (x, y)
    return pos


def get_line_graph_positions(SA, L=None, pos_G=None):
    """
    Positions for the line graph nodes: each node of ``L`` is an edge
    ``(u, v)`` of ``G`` (a nanowire), and it is placed at the **centroid** of
    the positions of its two endpoints.

    Parameters
    ----------
    SA : SkeletonAnalysis
        Already processed instance (with ``G``, ``L``, ``node_coords``, ``electrode_pos``).
    L : nx.Graph, optional
        Line graph to position. Defaults to ``SA.L``.
    pos_G : dict, optional
        Precomputed positions of ``G`` (to avoid recomputing them). By default
        they are obtained with ``get_node_positions(SA)``.

    Returns
    -------
    dict ``{(u, v): (x, y)}`` ready to pass to ``nx.draw_networkx_*``.
    """
    if L is None:
        L = SA.L
    if pos_G is None:
        pos_G = get_node_positions(SA)

    pos_L = {}
    for e in L.nodes:
        u, v = e[0], e[1]          # e may carry an extra key if G is a multigraph
        (xu, yu), (xv, yv) = pos_G[u], pos_G[v]
        pos_L[e] = ((xu + xv) / 2.0, (yu + yv) / 2.0)
    return pos_L


def plot_line_graph_on_image(SA, L=None, xlim=None, ylim=None, transparent=False,
                             save_path=None, dpi=200, ax=None,
                             node_size=25, edge_color="orange", edge_width=1.0,
                             background="skeleton", bg_cmap="gray", bg_alpha=1.0,
                             zoom_scale="auto"):
    """
    Draw the line graph on top of the image, like ``plot_network_on_image``
    but with nodes at the centroid of each edge of ``G``.

    Nodes of ``L`` touching an electrode are highlighted: red if one of their
    endpoints is a ``Vin_``, blue if it is a ``Vout_``.

    Parameters
    ----------
    SA : SkeletonAnalysis
        Already processed instance (with ``G``, ``L``, ``node_coords``, ``electrode_pos``, ``img``).
    L : nx.Graph, optional
        Line graph to draw. Defaults to ``SA.L``.
    xlim, ylim : tuple, optional
        Limits to zoom into a region.
    transparent : bool
        If True, hides the background image and makes the figure transparent.
    save_path : str, optional
        If given, saves the figure.
    dpi : int
        Resolution when saving.
    ax : matplotlib Axes, optional
        Axes to draw on; if not given, a new one is created.
    node_size, edge_width :
        Base size of nodes and edges at zoom 1 (full image).
    edge_color :
        Color of the edges of ``L``.
    background : {"skeleton", "img", "raw", "original"}, path, None or ndarray
        Background image. Defaults to the skeleton (``SA.skeleton_img``).
        See ``_get_background``.
    bg_cmap : str
        Background colormap ("gray", "gray_r", "bone", ...).
    bg_alpha : float
        Background opacity; lowering it helps the graph stand out over the
        original micrograph.
    zoom_scale : "auto", None or float
        Scaling of nodes/edges with zoom. "auto" enlarges them according to how
        much xlim/ylim shrink the visible region (nodes ~ f, edges ~ sqrt(f),
        with f clipped at 8x); None keeps them fixed; a number forces the factor.
    """
    if L is None:
        L = SA.L

    pos = get_line_graph_positions(SA, L=L)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    bg = _get_background(SA, background)
    if bg is not None:
        ax.imshow(bg, cmap=bg_cmap, alpha=0 if transparent else bg_alpha)

    def _touches(e, prefix):
        return any(_is_electrode(n, prefix) for n in (e[0], e[1]))

    vin_nodes = [e for e in L.nodes if _touches(e, "Vin_")]
    vout_nodes = [e for e in L.nodes if _touches(e, "Vout_")]
    network_nodes = [e for e in L.nodes if not (_touches(e, "Vin_") or _touches(e, "Vout_"))]

    f = _zoom_factor(SA, xlim, ylim, zoom_scale)
    ns, ne, nw = node_size * f, 150 * f, edge_width * f ** 0.5

    nx.draw_networkx_edges(L, pos, ax=ax, width=nw, edge_color=edge_color)
    nx.draw_networkx_nodes(L, pos, ax=ax, nodelist=network_nodes, node_size=ns, node_color="C0")
    nx.draw_networkx_nodes(L, pos, ax=ax, nodelist=vin_nodes, node_size=ne, node_color="red")
    nx.draw_networkx_nodes(L, pos, ax=ax, nodelist=vout_nodes, node_size=ne, node_color="blue")

    ax.axis("off")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if transparent:
        ax.figure.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    return ax

"""
preprocessing
=============

Preprocessing of nanowire photomicrographs prior to skeleton extraction
(``SkeletonAnalysis``).
"""

import cv2
import numpy as np


def load_gray(path):
    """Load an image in grayscale (uint8)."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def adaptive_binarize(img, block_size=151, C=15, intensity_thresh=55):
    """
    Gaussian adaptive binarization.

    Applies an adaptive threshold (local Gaussian-weighted mean over blocks)
    with replicated padding to avoid border artifacts, and additionally forces
    pixels below ``intensity_thresh`` to background (0).

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (uint8).
    block_size : int (odd)
        Size of the local window of the adaptive threshold.
    C : int
        Constant subtracted from the local mean.
    intensity_thresh : int
        Minimum intensity; pixels below it are forced to background.

    Returns
    -------
    np.ndarray : binary mask (0 / 255).
    """
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd.")

    pad = block_size // 2
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    mask_padded = cv2.adaptiveThreshold(
        img_padded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C,
    )
    mask = mask_padded[pad:-pad, pad:-pad]
    mask[img < intensity_thresh] = 0
    return mask


def overlay_mask(base_gray, binary_mask, color_rgb=(220, 20, 60)):
    """
    Overlay a binary mask in color on top of a grayscale image.
    Pixels with ``mask < 127`` are considered 'object'.

    Returns an RGB image (np.ndarray) to visually check the binarization
    against the original image.
    """
    base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
    base_rgb[binary_mask < 127] = color_rgb
    return base_rgb

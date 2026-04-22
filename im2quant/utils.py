"""Utility functions for im2quant."""

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import find_peaks


def crop_central_line(img: Image.Image, crop_half_frac: float = 0.20) -> Image.Image:
    """
    Auto-detect the central wire and crop around it.

    Strategy
    --------
    1. Compute row-wise mean brightness — wires appear as dark bands.
    2. Find all wire rows using peak detection on the *inverted* profile.
    3. Select the wire CLOSEST TO IMAGE CENTRE (not middle by index).
       ``wire_row = peaks[np.argmin(np.abs(peaks - h // 2))]``
       This is robust when the camera captures unequal substrate above/below.
    4. Crop ± crop_half_frac × H around the detected wire.

    Fallback
    --------
    If no peaks are found, falls back to a fixed 30/70 crop.
    """
    img_np = np.array(img.convert("RGB"))
    h, w = img_np.shape[:2]

    row_profile = np.mean(img_np, axis=(1, 2))  # [H]

    # Invert: wires are dark → peaks in inverted profile
    inv = row_profile.max() - row_profile

    peaks, _ = find_peaks(
        inv,
        height=inv.max() * 0.30,       # at least 30 % of max dip
        distance=int(h * 0.08),        # wires at least 8 % of height apart
        prominence=8,                  # must stand out from background
    )

    if len(peaks) == 0:
        # Fallback: fixed crop (30 % top, 30 % bottom)
        y1 = int(h * 0.30)
        y2 = int(h * 0.70)
        return img.crop((0, y1, w, y2))

    # Pick the wire closest to the image centre
    img_centre = h // 2
    wire_row = peaks[np.argmin(np.abs(peaks - img_centre))]

    crop_half = int(h * crop_half_frac)
    y1 = max(0, wire_row - crop_half)
    y2 = min(h, wire_row + crop_half)
    return img.crop((0, y1, w, y2))


def parse_batch_sample(filename: str):
    """
    Parse a filename like ``batch3.1.jpg`` → ``(3, 1)``.

    Returns
    -------
    (batch_id, sample_id) tuple of ints, or ``None`` if the filename does
    not match the expected pattern.
    """
    m = re.match(r"batch(\d+)\.(\d+)$", Path(filename).stem, re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else None



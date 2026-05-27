from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _smooth_counts(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _span_from_counts(counts: np.ndarray, threshold: float) -> tuple[int, int]:
    hits = np.where(counts > threshold)[0]
    if len(hits) == 0:
        return 0, len(counts) - 1
    return int(hits[0]), int(hits[-1])


def _is_tissue_like_rgb(rgb: np.ndarray) -> bool:
    r, g, b = [int(x) for x in rgb]
    sat = max(r, g, b) - min(r, g, b)
    return r > 95 and r - g > 8 and r - b > 18 and sat > 28


def _frame_candidate_mask(region: np.ndarray) -> np.ndarray:
    r = region[:, :, 0].astype(np.int16)
    g = region[:, :, 1].astype(np.int16)
    b = region[:, :, 2].astype(np.int16)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    sat = region.max(axis=2) - region.min(axis=2)
    tissue_like = (r > 95) & (r - g > 8) & (r - b > 18) & (sat > 28)
    blue_like = (b > 90) & (b - r > 20) & (b - g > 5)
    neutral_or_bright_ui = (luma > 55) & ~tissue_like & ((sat < 95) | (luma > 175))
    return blue_like | neutral_or_bright_ui


def _blue_frame_mask(region: np.ndarray) -> np.ndarray:
    r = region[:, :, 0].astype(np.int16)
    g = region[:, :, 1].astype(np.int16)
    b = region[:, :, 2].astype(np.int16)
    return (b > 90) & (b - r > 20) & (b - g > 5)


def _fixed_color_stripe_score(arr: np.ndarray, *, vertical: bool) -> float:
    h, w = arr.shape[:2]
    max_dim = max(h, w)
    step = max(1, int(max_dim // 900))
    small = arr[::step, ::step, :].astype(np.int16)
    h, w = small.shape[:2]
    luma = 0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]
    quantized = (small // 24).clip(0, 10)
    color_code = (
        quantized[:, :, 0] * 121 + quantized[:, :, 1] * 11 + quantized[:, :, 2]
    ).astype(np.int16)
    valid = luma > 45

    n_lines = w if vertical else h
    line_len = h if vertical else w
    offset = max(3, int(n_lines * 0.012))
    best = 0.0

    for i in range(offset, n_lines - offset):
        line_codes = color_code[:, i] if vertical else color_code[i, :]
        line_valid = valid[:, i] if vertical else valid[i, :]
        if line_valid.mean() < 0.25:
            continue

        counts = np.bincount(line_codes[line_valid], minlength=1331)
        dominant_code = int(counts.argmax())
        dominant_fraction = counts[dominant_code] / line_len
        if dominant_fraction < 0.10:
            continue

        qr = dominant_code // 121
        qg = (dominant_code % 121) // 11
        qb = dominant_code % 11
        rgb = np.array([qr * 24 + 12, qg * 24 + 12, qb * 24 + 12])
        line_luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        if line_luma < 48 or _is_tissue_like_rgb(rgb):
            continue

        before = color_code[:, i - offset] if vertical else color_code[i - offset, :]
        after = color_code[:, i + offset] if vertical else color_code[i + offset, :]
        neighbor_fraction = max((before == dominant_code).mean(), (after == dominant_code).mean())
        isolation = dominant_fraction - neighbor_fraction
        if isolation < 0.07:
            continue

        line = small[:, i, :] if vertical else small[i, :, :]
        before_rgb = small[:, i - offset, :] if vertical else small[i - offset, :, :]
        after_rgb = small[:, i + offset, :] if vertical else small[i + offset, :, :]
        contrast = max(
            float(np.linalg.norm(line.mean(axis=0) - before_rgb.mean(axis=0))),
            float(np.linalg.norm(line.mean(axis=0) - after_rgb.mean(axis=0))),
        )
        if contrast < 14:
            continue

        best = max(best, min(dominant_fraction, isolation + 0.12))

    return float(best)


def screen_artifact_signals(img: Image.Image) -> dict[str, float]:
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w = arr.shape[:2]
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    blue = (b > 90) & (b - r > 20) & (b - g > 5)
    return {
        "stripe_col": _fixed_color_stripe_score(arr, vertical=True),
        "stripe_row": _fixed_color_stripe_score(arr, vertical=False),
        "blue_col": float(blue.sum(axis=0).max() / max(h, 1)),
        "blue_row": float(blue.sum(axis=1).max() / max(w, 1)),
    }


def classify_screen_photo(img: Image.Image) -> tuple[bool, dict[str, Any]]:
    """Return (needs_precrop, signals_dict).

    Signals include stripe_col, stripe_row, blue_col, blue_row, and a ``reason``
    field listing which individual thresholds were exceeded.
    """
    signals = screen_artifact_signals(img)
    reasons = []
    if signals["stripe_col"] > 0.24:
        reasons.append("stripe_col")
    if signals["stripe_row"] > 0.24:
        reasons.append("stripe_row")
    if signals["blue_col"] > 0.06:
        reasons.append("blue_col")
    if signals["blue_row"] > 0.06:
        reasons.append("blue_row")
    needs_crop = len(reasons) > 0
    if not reasons:
        reasons.append("none")
    return needs_crop, {**signals, "reason": reasons}


def _detect_tissue_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    maxc = arr.max(axis=2).astype(np.int16)
    minc = arr.min(axis=2).astype(np.int16)

    tissue = (
        (r > 110)
        & (r - g > 12)
        & (r - b > 35)
        & (maxc - minc > 35)
    )

    # Most landscape originals have a left software sidebar and monitor bezel.
    # Portrait originals can have the image flush with the left edge, so do not
    # mask those away.
    if w > h:
        tissue[:, : int(w * 0.36)] = False

    col_counts = tissue.sum(axis=0)
    row_counts = tissue.sum(axis=1)
    col_s = _smooth_counts(col_counts, max(15, w * 0.018))
    row_s = _smooth_counts(row_counts, max(15, h * 0.018))

    x0, x1 = _span_from_counts(col_s, max(12, h * 0.012, col_s.max() * 0.06))
    y0, y1 = _span_from_counts(row_s, max(12, w * 0.012, row_s.max() * 0.06))

    tissue_w = max(1, x1 - x0 + 1)
    tissue_h = max(1, y1 - y0 + 1)
    pad_x_left = int(tissue_w * 0.018)
    pad_x_right = int(tissue_w * 0.075)
    pad_y_top = int(tissue_h * 0.035)
    pad_y_bottom = int(tissue_h * 0.055)

    x0 = max(0, x0 - pad_x_left)
    x1 = min(w - 1, x1 + pad_x_right)
    y0 = max(0, y0 - pad_y_top)
    y1 = min(h - 1, y1 + pad_y_bottom)

    # Do not let small red report text define a tiny crop. Fall back to the
    # usual screen-photo crop proportions if detection is implausible.
    if (x1 - x0) < w * 0.35 or (y1 - y0) < h * 0.35:
        if w > h:
            x0, x1 = int(w * 0.36), int(w * 0.965)
            y0, y1 = 0, h - 1
        else:
            x0, x1 = 0, int(w * 0.88)
            y0, y1 = int(h * 0.42), int(h * 0.98)

    return x0, y0, x1 + 1, y1 + 1


def _refine_to_window_frame(
    img: Image.Image, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w < 200 or crop_h < 200:
        return box
    crop_mask = _blue_frame_mask(arr[y0:y1, x0:x1])
    mask_fn = _frame_candidate_mask
    if crop_mask.sum(axis=0).max() > max(45, crop_h * 0.035) or crop_mask.sum(axis=1).max() > max(45, crop_w * 0.12):
        mask_fn = _blue_frame_mask

    scan_w = min(900, max(120, int(crop_w * 0.35)))
    left_region = arr[y0:y1, x0 : min(x1, x0 + scan_w)]
    left_blue = mask_fn(left_region).sum(axis=0)
    left_s = _smooth_counts(left_blue, 15)
    left_peak = int(left_s.argmax())
    if left_s[left_peak] > max(45, crop_h * 0.10):
        x0 = min(x1 - 80, x0 + left_peak + 3)

    crop_w = x1 - x0
    scan_w = min(900, max(120, int(crop_w * 0.35)))
    right_start = max(x0, x1 - scan_w)
    right_region = arr[y0:y1, right_start:x1]
    right_blue = mask_fn(right_region).sum(axis=0)
    right_s = _smooth_counts(right_blue, 15)
    right_peak = int(right_s.argmax())
    right_abs = right_start + right_peak
    if right_s[right_peak] > max(45, crop_h * 0.035) and right_abs > x0 + crop_w * 0.65:
        x1 = max(x0 + 80, min(x1, right_abs + 8))

    crop_w = x1 - x0
    crop_h = y1 - y0
    scan_h = min(700, max(100, int(crop_h * 0.25)))
    top_region = arr[y0 : min(y1, y0 + scan_h), x0:x1]
    top_blue = mask_fn(top_region).sum(axis=1)
    top_s = _smooth_counts(top_blue, 15)
    top_peak = int(top_s.argmax())
    if top_s[top_peak] > max(45, crop_w * 0.18):
        y0 = min(y1 - 80, y0 + top_peak + 3)

    crop_h = y1 - y0
    scan_h = min(700, max(100, int(crop_h * 0.25)))
    bottom_start = max(y0, y1 - scan_h)
    bottom_region = arr[bottom_start:y1, x0:x1]
    bottom_blue = mask_fn(bottom_region).sum(axis=1)
    bottom_s = _smooth_counts(bottom_blue, 15)
    bottom_peak = int(bottom_s.argmax())
    bottom_abs = bottom_start + bottom_peak
    if bottom_s[bottom_peak] > max(45, crop_w * 0.12) and bottom_abs > y0 + crop_h * 0.55:
        y1 = max(y0 + 80, min(y1, bottom_abs + 8))

    crop = arr[y0:y1, x0:x1]
    if h > w and crop.shape[0] > 120 and crop.shape[1] > 120:
        r = crop[:, :, 0]
        g = crop[:, :, 1]
        b = crop[:, :, 2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        sat = crop.max(axis=2) - crop.min(axis=2)
        light_ui = ((luma > 150) & (sat < 45)).mean(axis=1)
        light_s = _smooth_counts(light_ui, 21)
        tail = light_s[int(len(light_s) * 0.75) :]
        if len(tail) and tail[-30:].mean() > 0.35:
            candidates = np.where(light_s < 0.28)[0]
            candidates = candidates[candidates < int(len(light_s) * 0.9)]
            if len(candidates):
                cut = int(candidates[-1] + 4)
                if cut > len(light_s) * 0.45:
                    y1 = max(y0 + 80, y0 + cut)

    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _trim_light_ui_edges(
    img: Image.Image, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    crop = arr[y0:y1, x0:x1]
    if crop.shape[0] < 120 or crop.shape[1] < 120:
        return box

    r = crop[:, :, 0]
    g = crop[:, :, 1]
    b = crop[:, :, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    sat = crop.max(axis=2) - crop.min(axis=2)
    light_ui = (luma > 145) & (sat < 85)

    cw = x1 - x0
    ch = y1 - y0
    col_s = _smooth_counts(light_ui.mean(axis=0), max(7, cw * 0.006))
    row_s = _smooth_counts(light_ui.mean(axis=1), max(7, ch * 0.006))

    def _trim_end(signal: np.ndarray, length: int) -> int | None:
        edge = max(5, int(length * 0.01))
        if signal[-edge:].mean() < 0.42:
            return None
        i = len(signal) - 1
        while i > 0 and signal[i] > 0.42:
            i -= 1
        strip = len(signal) - 1 - i
        if int(length * 0.015) <= strip <= int(length * 0.22):
            return i + 1
        return None

    right_cut = _trim_end(col_s, cw)
    if right_cut is not None:
        x1 = max(x0 + 80, x0 + right_cut)

    bottom_cut = _trim_end(row_s, ch)
    if bottom_cut is not None:
        y1 = max(y0 + 80, y0 + bottom_cut)

    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def crop_screen_photo_window(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop a phone-photo-of-screen image down to the laryngoscope window.

    Returns ``(cropped_image, (x0, y0, x1, y1))`` where the box coordinates
    are in the original image pixel space.
    """
    img = img.convert("RGB")
    box = _detect_tissue_bbox(img)
    box = _refine_to_window_frame(img, box)
    box = _trim_light_ui_edges(img, box)
    cropped = img.crop(box)
    return cropped, box

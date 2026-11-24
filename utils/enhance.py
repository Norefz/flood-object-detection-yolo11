
"""Image enhancement utilities for BGR images."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def apply_enhancement(img_bgr, method="none", brightness=0):
    """Terapkan enhancement pada gambar BGR dan kembalikan gambar BGR."""
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        raise ValueError("img_bgr harus berupa array gambar OpenCV.")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr harus memiliki tiga channel BGR.")
    if img_bgr.dtype != np.uint8:
        raise ValueError("img_bgr harus bertipe uint8.")

    method = str(method).lower().strip()
    if method not in {"none", "he", "clahe", "cs", "gamma"}:
        raise ValueError("method harus none, he, clahe, cs, atau gamma.")

    if method == "none":
        enhanced = img_bgr.copy()
    elif method == "he":
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        y_channel = cv2.equalizeHist(y_channel)
        enhanced = cv2.cvtColor(
            cv2.merge((y_channel, cr_channel, cb_channel)), cv2.COLOR_YCrCb2BGR
        )
    elif method == "clahe":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
        )
    elif method == "cs":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        low, high = np.percentile(l_channel, (2, 98))
        if high > low:
            stretched = (l_channel.astype(np.float32) - low) * (255.0 / (high - low))
            l_channel = np.clip(stretched, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
        )
    else:  # Gamma 0.6 mencerahkan nilai piksel gelap.
        gamma = 0.6
        lookup = np.rint(((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255)
        enhanced = cv2.LUT(img_bgr, lookup.astype(np.uint8))

    try:
        brightness = float(np.clip(float(brightness), -100, 100))
    except (TypeError, ValueError) as error:
        raise ValueError("brightness harus berupa angka dari -100 sampai 100.") from error

    return cv2.convertScaleAbs(enhanced, alpha=1.0, beta=brightness)


def _make_labeled_panel(image, label, max_width=480):
    """Skalakan gambar seperlunya dan tambahkan label pada bilah atas."""
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width)
    panel_width = max(1, int(round(width * scale)))
    panel_height = max(1, int(round(height * scale)))
    if scale < 1.0:
        image = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_AREA)

    header = np.full((36, panel_width, 3), (27, 35, 48), dtype=np.uint8)
    cv2.putText(
        header,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def _save_comparison(input_path, output_path, brightness=0):
    """Simpan gambar asli dan seluruh metode dalam satu baris perbandingan."""
    original = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Gambar tidak dapat dibaca: {input_path}")

    # EqualizeHist langsung pada BGR memperlakukan kanal warna secara terpisah,
    # sehingga hue dan saturasi dapat bergeser; karena itu HE hanya mengubah luminance.
    panels = [_make_labeled_panel(original, "Asli")]
    for method, label in (
        ("none", "None"),
        ("he", "HE"),
        ("clahe", "CLAHE"),
        ("cs", "CS"),
        ("gamma", "Gamma 0.6"),
    ):
        result = apply_enhancement(original, method=method, brightness=brightness)
        panels.append(_make_labeled_panel(result, label))

    comparison = np.hstack(panels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), comparison):
        raise OSError(f"Gambar perbandingan gagal disimpan: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simpan perbandingan enhancement berdampingan untuk satu gambar."
    )
    parser.add_argument("image", help="Path gambar input.")
    parser.add_argument(
        "--output",
        help="Path gambar hasil perbandingan (default: <nama>_enhancement_comparison.jpg).",
    )
    parser.add_argument("--brightness", type=int, choices=range(-100, 101), default=0)
    args = parser.parse_args()

    source_path = Path(args.image)
    output_path = Path(args.output) if args.output else source_path.with_name(
        f"{source_path.stem}_enhancement_comparison.jpg"
    )
    try:
        saved_path = _save_comparison(source_path, output_path, args.brightness)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Gambar perbandingan disimpan: {saved_path}")

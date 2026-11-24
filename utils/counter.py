"""Utilitas geometri region dan anotasi deteksi kendaraan."""

import cv2
import numpy as np


def region_to_pixels(region_norm, w, h):
    """Konversi poligon ternormalisasi ke koordinat piksel, atau None."""
    if region_norm is None or len(region_norm) < 3 or w <= 0 or h <= 0:
        return None

    try:
        points = np.asarray(region_norm, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return None

    # Jaga titik tetap di dalam gambar sebelum mengubahnya ke piksel.
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * (w - 1)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * (h - 1)
    return np.rint(points).astype(np.int32)


def point_in_region(poly, cx, cy):
    """Periksa apakah titik berada di dalam atau pada garis poligon."""
    if poly is None or len(poly) < 3:
        return False
    contour = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
    # Nilai nol berarti titik tepat pada batas; batas dihitung sebagai di dalam.
    return cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0


def draw_detections(img, boxes, poly):
    """Gambar kotak deteksi dan region; boxes berisi x1,y1,x2,y2,confidence."""
    output = img.copy()
    if poly is not None and len(poly) >= 3:
        contour = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
        overlay = output.copy()
        cv2.fillPoly(overlay, [contour], (44, 190, 130))
        output = cv2.addWeighted(overlay, 0.18, output, 0.82, 0)
        cv2.polylines(output, [contour], True, (44, 190, 130), 2, cv2.LINE_AA)

    for box in boxes:
        x1, y1, x2, y2 = (int(round(float(value))) for value in box[:4])
        confidence = float(box[4]) if len(box) > 4 else 0.0
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        inside = point_in_region(poly, center_x, center_y)
        color = (36, 180, 95) if inside else (220, 145, 48)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{confidence:.2f}"
        text_y = max(18, y1 - 7)
        cv2.putText(
            output, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, color, 2, cv2.LINE_AA
        )
    return output


def tracked_detections(result):
    """Ambil koordinat, confidence, dan ID dari satu hasil track YOLO."""
    tracked_boxes = getattr(result, "boxes", None) if result is not None else None
    if tracked_boxes is None:
        return []

    coordinates = tracked_boxes.xyxy.cpu().numpy()
    confidences = tracked_boxes.conf.cpu().numpy()
    raw_ids = tracked_boxes.id
    ids = raw_ids.cpu().numpy() if raw_ids is not None else [None] * len(coordinates)

    detections = []
    for box, confidence, track_id in zip(coordinates, confidences, ids):
        # ID dapat None ketika tracker belum memiliki pasangan track pada frame.
        normalized_id = int(track_id) if track_id is not None else None
        detections.append((*[float(value) for value in box[:4]], float(confidence), normalized_id))
    return detections


def update_track_counts(detections, poly, seen_ids, entered_ids):
    """Perbarui himpunan ID unik dan hitung deteksi yang sedang di region."""
    inside_now = 0
    for x1, y1, x2, y2, _confidence, track_id in detections:
        inside = point_in_region(poly, (x1 + x2) / 2, (y1 + y2) / 2)
        inside_now += int(inside)
        if track_id is not None:
            seen_ids.add(track_id)
            if inside:
                entered_ids.add(track_id)
    return inside_now


def draw_tracking_frame(img, detections, poly, stats):
    """Gambar kotak ber-ID dan tiga statistik di pojok kiri atas frame."""
    output = draw_detections(img, [detection[:5] for detection in detections], poly)
    for x1, y1, _x2, _y2, _confidence, track_id in detections:
        if track_id is not None:
            cv2.putText(
                output,
                f"ID {track_id}",
                (int(round(x1)), max(34, int(round(y1)) + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

    # Panel transparan menjaga angka terbaca di atas latar video yang beragam.
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (285, 105), (18, 27, 38), -1)
    output = cv2.addWeighted(overlay, 0.72, output, 0.28, 0)
    labels = (
        f"Di region sekarang: {stats['inside_now']}",
        f"Pernah masuk region: {stats['entered_region']}",
        f"Total ID unik: {stats['unique_total']}",
    )
    for index, label in enumerate(labels):
        cv2.putText(
            output,
            label,
            (20, 34 + index * 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output

"""Server Flask dan manajemen model YOLO."""

import io
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from utils.counter import (
    draw_detections,
    draw_tracking_frame,
    point_in_region,
    region_to_pixels,
    tracked_detections,
    update_track_counts,
)
from utils.enhance import apply_enhancement


# Siapkan path proyek agar server bisa dijalankan dari direktori mana pun.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
REPORT_LOG = REPORT_DIR / "report_log.json"
SETTINGS_FILE = BASE_DIR / "settings.json"
MAX_MODEL_SIZE = 200 * 1024 * 1024
MAX_UPLOAD_SIZE = 500 * 1024 * 1024
FILE_RETENTION_SECONDS = 24 * 60 * 60
FALLBACK_MODEL_NAME = "yolo11n.pt"
FALLBACK_CLASSES = [1, 2, 3, 5, 7]
DEFAULT_SETTINGS = {"active_model": None, "conf": 0.25}

# Gunakan lock terpisah untuk pergantian model dan penulisan pengaturan.
MODEL_LOCK = threading.Lock()
SETTINGS_LOCK = threading.Lock()
REPORT_LOCK = threading.Lock()
VIDEO_JOBS_LOCK = threading.Lock()
VIDEO_WORKER_LOCK = threading.Lock()
VIDEO_JOBS = {}
VIDEO_QUEUE = queue.Queue()
VIDEO_WORKER_STARTED = False

# Simpan state aktif di memori untuk dipakai route berikutnya.
ACTIVE_MODEL = None
ACTIVE_MODEL_NAME = None
ACTIVE_MODEL_IS_FALLBACK = False
ACTIVE_CLASSES = None
SETTINGS = dict(DEFAULT_SETTINGS)

# Batasi ukuran keseluruhan request upload menjadi 500 MB.
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yolo_model(model_path):
    """Muat bobot dengan Ultralytics; dipisah agar proses uji bisa mengganti loader."""
    from ultralytics import YOLO

    return YOLO(str(model_path))


def _valid_conf(value):
    """Pastikan confidence berada pada rentang slider aplikasi."""
    if isinstance(value, bool):
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(conf) or not 0.1 <= conf <= 0.9:
        return None
    return conf


def _cleanup_old_files():
    """Hapus file upload/hasil yang berumur lebih dari 24 jam."""
    cutoff = time.time() - FILE_RETENTION_SECONDS
    removed = 0
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            for path in folder.rglob("*"):
                # Pertahankan penanda folder kosong yang dilacak Git.
                if path.name == ".gitkeep" or not path.is_file():
                    continue
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError as error:
                    logger.warning("File lama gagal dibersihkan (%s): %s", path, error)
        except OSError as error:
            logger.warning("Folder sementara gagal diperiksa (%s): %s", folder, error)
    if removed:
        logger.info("Pembersihan startup menghapus %s file berumur lebih dari 24 jam.", removed)


def _read_report_rows():
    """Baca log laporan; file kosong, rusak, atau belum ada dianggap tanpa baris."""
    try:
        if not REPORT_LOG.exists() or REPORT_LOG.stat().st_size == 0:
            return []
        with REPORT_LOG.open("r", encoding="utf-8") as report_file:
            rows = json.load(report_file)
        if not isinstance(rows, list):
            raise ValueError("akar JSON laporan harus berupa array")
        return [row for row in rows if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        logger.warning("Log laporan tidak dapat dibaca; gunakan daftar kosong: %s", error)
        return []


def _write_report_rows_atomic(rows):
    """Ganti report_log.json secara atomik agar log tidak terpotong saat ditulis."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=".report-", suffix=".tmp", dir=REPORT_DIR
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            json.dump(rows, temp_file, ensure_ascii=False, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, REPORT_LOG)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def _make_enhancement_report(file_name, kind, model_name, method, brightness, conf, baseline, result):
    """Satukan angka sebelum/sesudah dan hitung selisih sesudah dikurangi sebelum."""
    before_total = int(baseline["total"])
    after_total = int(result["total"])
    before_region = int(baseline["inside_region"])
    after_region = int(result["inside_region"])
    before_conf = float(baseline["avg_conf"])
    after_conf = float(result["avg_conf"])
    return {
        "waktu": datetime.now().astimezone().isoformat(timespec="seconds"),
        "nama_file": file_name,
        "jenis": kind,
        "model": model_name,
        "metode_enhancement": method,
        "brightness": float(brightness),
        "conf": float(conf),
        "total_sebelum": before_total,
        "total_sesudah": after_total,
        "selisih_total": after_total - before_total,
        "region_sebelum": before_region,
        "region_sesudah": after_region,
        "selisih_region": after_region - before_region,
        "avgconf_sebelum": before_conf,
        "avgconf_sesudah": after_conf,
        "selisih_avgconf": after_conf - before_conf,
    }


def _append_enhancement_report(file_name, kind, model_name, method, brightness, conf, baseline, result):
    """Tambahkan satu laporan baseline di bawah lock lalu simpan atomik."""
    row = _make_enhancement_report(
        file_name, kind, model_name, method, brightness, conf, baseline, result
    )
    with REPORT_LOCK:
        rows = _read_report_rows()
        rows.append(row)
        _write_report_rows_atomic(rows)
    return row


def _write_settings_atomic(settings):
    """Tulis JSON ke file sementara, lalu ganti file tujuan secara atomik."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        # Buat file sementara di folder yang sama agar os.replace tetap atomik.
        fd, temp_name = tempfile.mkstemp(
            prefix=".settings-", suffix=".tmp", dir=SETTINGS_FILE.parent
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            json.dump(settings, temp_file, ensure_ascii=False, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Ganti settings.json hanya setelah isi file sementara selesai ditulis.
        os.replace(temp_path, SETTINGS_FILE)
    finally:
        # Bersihkan file sementara jika proses tulis atau penggantian gagal.
        if temp_path and temp_path.exists():
            temp_path.unlink()


def _read_settings():
    """Baca settings.json dan gunakan nilai aman jika file rusak atau belum ada."""
    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as settings_file:
            stored = json.load(settings_file)
        if not isinstance(stored, dict):
            raise ValueError("settings.json harus berisi object JSON.")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        logger.warning("settings.json tidak dapat dibaca; gunakan nilai awal: %s", error)
        stored = dict(DEFAULT_SETTINGS)

    # Ambil hanya key kontrak dan validasi confidence yang tersimpan.
    conf = _valid_conf(stored.get("conf"))
    active_name = stored.get("active_model")
    if not isinstance(active_name, str):
        active_name = None
    normalized = {
        "active_model": active_name,
        "conf": conf if conf is not None else DEFAULT_SETTINGS["conf"],
    }

    # Buat file awal atau rapikan nilai rusak agar state disk sesuai kontrak.
    if not SETTINGS_FILE.exists() or normalized != stored:
        try:
            _write_settings_atomic(normalized)
        except OSError as error:
            logger.warning("settings.json gagal diperbarui: %s", error)
    return normalized


def _safe_model_name(name):
    """Terima nama file .pt polos tanpa path atau segmen direktori."""
    if not isinstance(name, str) or not name:
        return None
    safe_name = secure_filename(name)
    if safe_name != name or Path(safe_name).suffix.lower() != ".pt":
        return None
    return safe_name


def initialize_app_state():
    """Muat model tersimpan saat start, lalu fallback hanya bila env mengaktifkannya."""
    global ACTIVE_MODEL, ACTIVE_MODEL_NAME, ACTIVE_MODEL_IS_FALLBACK, ACTIVE_CLASSES, SETTINGS

    # Bersihkan file sementara lama sebelum server menerima permintaan.
    _cleanup_old_files()

    # Baca confidence dan nama model tersimpan lebih dulu.
    loaded_settings = _read_settings()
    SETTINGS = dict(loaded_settings)
    ACTIVE_MODEL = None
    ACTIVE_MODEL_NAME = None
    ACTIVE_MODEL_IS_FALLBACK = False
    ACTIVE_CLASSES = None

    # Muat model aktif hanya jika namanya aman dan file .pt benar-benar tersedia.
    stored_name = _safe_model_name(loaded_settings.get("active_model"))
    stored_path = MODEL_DIR / stored_name if stored_name else None
    if stored_path and stored_path.is_file():
        try:
            model = load_yolo_model(stored_path)
            with MODEL_LOCK:
                ACTIVE_MODEL = model
                ACTIVE_MODEL_NAME = stored_name
                ACTIVE_CLASSES = None
            return
        except Exception as error:
            logger.warning("Model tersimpan %s gagal dimuat: %s", stored_name, error)

    # Jika model pilihan hilang atau gagal dimuat, hapus referensinya dari settings.
    if loaded_settings.get("active_model") is not None:
        loaded_settings["active_model"] = None
        try:
            _write_settings_atomic(loaded_settings)
            SETTINGS = dict(loaded_settings)
        except OSError as error:
            logger.warning("Referensi model rusak gagal dibersihkan: %s", error)

    # Jangan ambil bobot fallback kecuali proses dijalankan dengan USE_FALLBACK=1.
    if os.environ.get("USE_FALLBACK") == "1":
        try:
            model = load_yolo_model(FALLBACK_MODEL_NAME)
            with MODEL_LOCK:
                ACTIVE_MODEL = model
                ACTIVE_MODEL_NAME = FALLBACK_MODEL_NAME
                ACTIVE_MODEL_IS_FALLBACK = True
                ACTIVE_CLASSES = list(FALLBACK_CLASSES)
        except Exception as error:
            logger.warning("Model fallback %s gagal dimuat: %s", FALLBACK_MODEL_NAME, error)


def _set_active_model(model, name, is_fallback=False):
    """Simpan nama aktif lalu ganti state; pemanggil wajib memegang MODEL_LOCK."""
    global ACTIVE_MODEL, ACTIVE_MODEL_NAME, ACTIVE_MODEL_IS_FALLBACK, ACTIVE_CLASSES, SETTINGS
    with SETTINGS_LOCK:
        updated = dict(SETTINGS)
        updated["active_model"] = None if is_fallback else name
        _write_settings_atomic(updated)
        SETTINGS = updated

    # Caller memegang MODEL_LOCK selama tiga nilai global ini diganti.
    ACTIVE_MODEL = model
    ACTIVE_MODEL_NAME = name
    ACTIVE_MODEL_IS_FALLBACK = is_fallback
    ACTIVE_CLASSES = list(FALLBACK_CLASSES) if is_fallback else None


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_error):
    """Kembalikan error JSON saat ukuran request melewati batas server."""
    return jsonify({"ok": False, "error": "Ukuran upload melebihi batas 500 MB."}), 413


@app.get("/")
def index():
    """Render aplikasi frontend."""
    return render_template("index.html")


@app.get("/api/models")
def get_models():
    """Daftarkan file .pt lokal dan state model aktif."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for model_path in sorted(MODEL_DIR.iterdir(), key=lambda path: path.name.lower()):
        if (
            not model_path.is_file()
            or model_path.name.startswith(".")
            or model_path.suffix.lower() != ".pt"
        ):
            continue
        stat = model_path.stat()
        models.append({
            "name": model_path.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "uploaded": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        })

    # Baca nilai aktif dan flag fallback dalam satu snapshot yang konsisten.
    with MODEL_LOCK:
        active = ACTIVE_MODEL_NAME
        fallback = ACTIVE_MODEL_IS_FALLBACK
    return jsonify({"models": models, "active": active, "fallback": fallback})


@app.post("/api/upload_model")
def upload_model():
    """Validasi, simpan sementara, muat, lalu aktifkan model upload."""
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"ok": False, "error": "Pilih file model .pt terlebih dahulu."}), 400

    # Bersihkan nama sebelum membentuk path di folder model.
    model_name = secure_filename(uploaded.filename)
    if not model_name or Path(model_name).suffix.lower() != ".pt":
        return jsonify({"ok": False, "error": "Format model harus .pt."}), 400

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = None
    final_path = MODEL_DIR / model_name
    final_created = False
    backup_path = None
    try:
        # Simpan upload ke file sementara agar file final hanya muncul setelah valid.
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".upload-", suffix=".pt", dir=MODEL_DIR, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            uploaded.save(temp_file)

        # Ukur file tersimpan, bukan hanya Content-Length multipart.
        if temp_path.stat().st_size > MAX_MODEL_SIZE:
            return jsonify({"ok": False, "error": "Ukuran model maksimal 200 MB."}), 413

        # Coba muat checkpoint sebelum menaruhnya sebagai model yang tersedia.
        try:
            model = load_yolo_model(temp_path)
        except Exception as error:
            return jsonify({"ok": False, "error": f"Model .pt tidak dapat dimuat: {error}"}), 400

        # NamedTemporaryFile biasanya membuat file berizin 0600. Model yang
        # tersimpan perlu dapat dibaca lagi oleh proses aplikasi setelah restart
        # atau saat dijalankan melalui service account yang berbeda.
        try:
            temp_path.chmod(0o644)
        except OSError as error:
            return jsonify({"ok": False, "error": f"Izin file model gagal disiapkan: {error}"}), 500

        # Serialisasi pergantian file, settings, dan state model aktif.
        with MODEL_LOCK:
            try:
                # Simpan versi lama sementara supaya bisa dipulihkan jika aktivasi gagal.
                if final_path.exists():
                    backup_path = MODEL_DIR / f".replace-{uuid.uuid4().hex}.bak"
                    os.replace(final_path, backup_path)

                # Ganti file model lama secara atomik dengan checkpoint yang sudah dimuat.
                os.replace(temp_path, final_path)
                temp_path = None
                final_created = True
                _set_active_model(model, model_name)
            except OSError as error:
                # Pulihkan file sebelumnya agar model yang lama tetap tersedia.
                if final_created:
                    final_path.unlink(missing_ok=True)
                if backup_path and backup_path.exists():
                    os.replace(backup_path, final_path)
                final_created = False
                return jsonify({"ok": False, "error": f"Gagal menyimpan pengaturan model: {error}"}), 500
            if backup_path:
                try:
                    backup_path.unlink(missing_ok=True)
                except OSError as error:
                    logger.warning("Cadangan model lama gagal dibersihkan (%s): %s", backup_path, error)

        return jsonify({"ok": True, "active": model_name})
    except OSError as error:
        # Jelaskan kegagalan disk dan bersihkan artefak parsial bila ada.
        if final_created and final_path.exists():
            final_path.unlink(missing_ok=True)
        return jsonify({"ok": False, "error": f"File model gagal disimpan: {error}"}), 500
    finally:
        # Hapus file upload sementara setelah sukses maupun gagal.
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as error:
                logger.warning("File sementara model gagal dibersihkan (%s): %s", temp_path, error)


@app.post("/api/select_model")
def select_model():
    """Muat dan aktifkan model .pt yang sudah tersimpan di folder model/."""
    data = request.get_json(silent=True) or {}
    model_name = _safe_model_name(data.get("name"))
    if not model_name:
        with MODEL_LOCK:
            active = ACTIVE_MODEL_NAME
        return jsonify({"ok": False, "active": active}), 400

    model_path = MODEL_DIR / model_name
    if not model_path.is_file():
        with MODEL_LOCK:
            active = ACTIVE_MODEL_NAME
        return jsonify({"ok": False, "active": active}), 404

    # Muat di luar lock agar pembaca state model tidak terblokir selama load.
    try:
        model = load_yolo_model(model_path)
    except Exception as error:
        logger.warning("Model %s gagal dimuat: %s", model_name, error)
        with MODEL_LOCK:
            active = ACTIVE_MODEL_NAME
        return jsonify({"ok": False, "active": active}), 400

    try:
        with MODEL_LOCK:
            # Pastikan file belum dihapus selama model sedang dimuat.
            if not model_path.is_file():
                return jsonify({"ok": False, "active": ACTIVE_MODEL_NAME}), 404
            _set_active_model(model, model_name)
    except OSError as error:
        logger.error("Settings gagal menyimpan model aktif: %s", error)
        with MODEL_LOCK:
            active = ACTIVE_MODEL_NAME
        return jsonify({"ok": False, "active": active}), 500
    return jsonify({"ok": True, "active": model_name})


@app.post("/api/delete_model")
def delete_model():
    """Hapus model lokal dan nonaktifkan lebih dulu bila model sedang aktif."""
    data = request.get_json(silent=True) or {}
    model_name = _safe_model_name(data.get("name"))
    if not model_name:
        return jsonify({"ok": False}), 400

    model_path = MODEL_DIR / model_name
    with MODEL_LOCK:
        if not model_path.is_file():
            return jsonify({"ok": False, "error": "Model tidak ditemukan."}), 404

        # Video memakai path bobot saat worker mulai memproses. Pertahankan file
        # sampai seluruh job yang memakai model tersebut selesai.
        with VIDEO_JOBS_LOCK:
            in_use = any(
                job.get("model_name") == model_name
                and job.get("status") in {"queued", "running"}
                for job in VIDEO_JOBS.values()
            )
        if in_use:
            return jsonify({
                "ok": False,
                "error": "Model sedang dipakai oleh pemrosesan video. Coba lagi setelah selesai.",
            }), 409

        if model_name == ACTIVE_MODEL_NAME:
            # Pindahkan dulu ke nama sementara. Jika settings gagal disimpan,
            # kembalikan file agar model aktif tetap utuh.
            backup_path = MODEL_DIR / f".delete-{uuid.uuid4().hex}.bak"
            try:
                os.replace(model_path, backup_path)
                _set_active_model(None, None)
            except OSError as error:
                if backup_path.exists():
                    os.replace(backup_path, model_path)
                logger.error("Model aktif %s gagal dinonaktifkan: %s", model_name, error)
                return jsonify({
                    "ok": False,
                    "error": f"Model gagal dinonaktifkan: {error}",
                }), 500
            try:
                backup_path.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("File model sementara gagal dibersihkan (%s): %s", backup_path, error)
            return jsonify({"ok": True, "active": None})

        try:
            model_path.unlink()
        except OSError as error:
            logger.error("Model %s gagal dihapus: %s", model_name, error)
            return jsonify({"ok": False, "error": f"Model gagal dihapus: {error}"}), 500
    return jsonify({"ok": True})


@app.get("/api/settings")
def get_settings():
    """Kirim confidence default dari state pengaturan."""
    with SETTINGS_LOCK:
        conf = SETTINGS["conf"]
    return jsonify({"conf": conf})


def _prediction_data(model, image, conf, poly):
    """Jalankan prediksi dan hitung statistik dari titik tengah tiap kotak."""
    predictions = model.predict(source=image, conf=conf, verbose=False)
    result = predictions[0] if predictions else None
    detected_boxes = getattr(result, "boxes", None) if result is not None else None
    if detected_boxes is None:
        xyxy = np.empty((0, 4), dtype=np.float32)
        confidences = np.empty((0,), dtype=np.float32)
    else:
        # Pindahkan tensor Ultralytics ke NumPy agar pemrosesan OpenCV konsisten.
        xyxy = detected_boxes.xyxy.cpu().numpy()
        confidences = detected_boxes.conf.cpu().numpy()

    boxes = []
    inside_region = 0
    for coordinates, confidence in zip(xyxy, confidences):
        x1, y1, x2, y2 = (float(value) for value in coordinates[:4])
        score = float(confidence)
        boxes.append((x1, y1, x2, y2, score))
        if point_in_region(poly, (x1 + x2) / 2, (y1 + y2) / 2):
            inside_region += 1

    average_confidence = float(np.mean(confidences)) if len(confidences) else 0.0
    return boxes, {
        "total": len(boxes),
        "inside_region": inside_region,
        "avg_conf": round(average_confidence, 6),
    }


@app.post("/api/detect_image")
def detect_image():
    """Validasi gambar, lakukan deteksi, lalu simpan tiga gambar hasil."""
    # Pastikan model aktif tersedia sebelum membaca dan memproses upload.
    with MODEL_LOCK:
        model = ACTIVE_MODEL
        model_name = ACTIVE_MODEL_NAME
        if model is None:
            return jsonify({"error": "Model belum tersedia. Upload atau pilih model .pt terlebih dahulu."}), 503

        # Baca upload sebagai gambar BGR dan tolak berkas rusak/non-gambar.
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Pilih file foto untuk dideteksi."}), 400
        raw_bytes = uploaded.read()
        image = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "File yang diunggah bukan gambar yang valid atau formatnya tidak didukung."}), 400

        # Ambil dan validasi parameter sesuai rentang kontrol frontend.
        with SETTINGS_LOCK:
            default_conf = SETTINGS["conf"]
        conf = _valid_conf(request.form.get("conf", default_conf))
        if conf is None:
            return jsonify({"error": "Confidence harus bernilai antara 0.1 dan 0.9."}), 400
        method = request.form.get("enhance", "none").strip().lower()
        if method not in {"none", "he", "clahe", "cs", "gamma"}:
            return jsonify({"error": "Metode enhancement tidak dikenal."}), 400
        try:
            brightness = float(request.form.get("brightness", 0))
            if not math.isfinite(brightness) or not -100 <= brightness <= 100:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "Brightness harus bernilai antara -100 dan 100."}), 400

        # Ubah region JSON ternormalisasi menjadi poligon koordinat piksel.
        region_value = request.form.get("region", "").strip()
        if region_value:
            try:
                region_norm = json.loads(region_value)
                if not isinstance(region_norm, list) or len(region_norm) < 3:
                    raise ValueError
                poly = region_to_pixels(region_norm, image.shape[1], image.shape[0])
                if poly is None:
                    raise ValueError
            except (json.JSONDecodeError, TypeError, ValueError):
                return jsonify({"error": "Region harus berupa poligon JSON dengan minimal tiga titik."}), 400
        else:
            poly = None

        # Jalankan baseline hanya ketika metode enhancement dipilih.
        baseline = None
        try:
            if method != "none":
                _, baseline = _prediction_data(model, image, conf, poly)

            # Terapkan enhancement, termasuk brightness, sebelum deteksi utama.
            enhanced = apply_enhancement(image, method=method, brightness=brightness)
            boxes, stats = _prediction_data(model, enhanced, conf, poly)
            rendered = draw_detections(enhanced, boxes, poly)
        except Exception as error:
            logger.exception("Deteksi foto gagal")
            return jsonify({"error": f"Deteksi gagal: {error}"}), 500

    # Simpan semua keluaran dengan satu ID unik agar URL mudah ditautkan.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_id = uuid.uuid4().hex
    filenames = {
        "original": f"{output_id}_original.jpg",
        "enhanced": f"{output_id}_enhanced.jpg",
        "result": f"{output_id}_result.jpg",
    }
    images = {"original": image, "enhanced": enhanced, "result": rendered}
    try:
        for key, filename in filenames.items():
            if not cv2.imwrite(str(OUTPUT_DIR / filename), images[key]):
                raise OSError(f"Gambar {key} gagal disimpan.")
    except OSError as error:
        # Hapus file yang sempat tersimpan jika salah satu penulisan gagal.
        for filename in filenames.values():
            (OUTPUT_DIR / filename).unlink(missing_ok=True)
        return jsonify({"error": f"Hasil deteksi gagal disimpan: {error}"}), 500

    # Catat hanya deteksi yang benar-benar memiliki hasil baseline.
    if baseline is not None:
        try:
            _append_enhancement_report(
                secure_filename(uploaded.filename) or uploaded.filename,
                "foto",
                model_name,
                method,
                brightness,
                conf,
                baseline,
                stats,
            )
        except OSError as error:
            logger.exception("Laporan deteksi foto gagal disimpan")
            return jsonify({"error": f"Laporan deteksi gagal disimpan: {error}"}), 500

    return jsonify({
        "original_url": f"/outputs/{filenames['original']}",
        "enhanced_url": f"/outputs/{filenames['enhanced']}",
        "result_url": f"/outputs/{filenames['result']}",
        **stats,
        "baseline": baseline,
        "model": model_name,
    })


def _set_video_job(job_id, **changes):
    """Perbarui state job di bawah lock agar polling membaca snapshot konsisten."""
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if job is not None:
            job.update(changes)


def _video_stats(inside_now, entered_ids, seen_ids):
    """Bangun statistik frame sesuai respons endpoint progress."""
    return {
        "inside_now": int(inside_now),
        "entered_region": len(entered_ids),
        "unique_total": len(seen_ids),
    }


def _run_video_pass(job, model_source, frame_size, fps, frame_total, pass_index, pass_total, write_path=None):
    """Jalankan satu lintasan video; baseline hanya menghitung, lintasan utama menulis."""
    capture = cv2.VideoCapture(str(job["input_path"]))
    if not capture.isOpened():
        capture.release()
        raise ValueError("File video tidak dapat dibuka atau codec-nya tidak didukung.")

    writer = None
    seen_ids = set()
    entered_ids = set()
    confidence_values = []
    last_stats = _video_stats(0, entered_ids, seen_ids)
    processed = 0
    try:
        # Muat model setelah capture masuk try agar error tetap melepas file video.
        model = load_yolo_model(model_source)
        # Buat writer hanya pada lintasan hasil yang akan ditonton pengguna.
        if write_path is not None:
            writer = cv2.VideoWriter(
                str(write_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                frame_size,
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV gagal membuat video sementara.")

        while True:
            success, frame = capture.read()
            if not success:
                break

            # Samakan ukuran frame dan batasi lebar keluaran maksimal 960 piksel.
            if frame.shape[1] > 960:
                height = max(1, round(frame.shape[0] * 960 / frame.shape[1]))
                frame = cv2.resize(frame, (960, height), interpolation=cv2.INTER_AREA)
            if (frame.shape[1], frame.shape[0]) != frame_size:
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

            # Baseline memakai frame mentah; lintasan utama memakai enhancement.
            source_frame = frame
            if write_path is not None:
                source_frame = apply_enhancement(
                    frame, method=job["enhance"], brightness=job["brightness"]
                )

            # persist menjaga ID ByteTrack konsisten antar-frame dalam satu video.
            predictions = model.track(
                source=source_frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=job["conf"],
                verbose=False,
            )
            result = predictions[0] if predictions else None
            detections = tracked_detections(result)
            poly = region_to_pixels(job["region"], frame_size[0], frame_size[1])
            inside_now = update_track_counts(detections, poly, seen_ids, entered_ids)
            confidence_values.extend(detection[4] for detection in detections)
            last_stats = _video_stats(inside_now, entered_ids, seen_ids)

            # Tulis video hanya untuk lintasan enhancement yang menjadi hasil akhir.
            if writer is not None:
                annotated = draw_tracking_frame(source_frame, detections, poly, last_stats)
                writer.write(annotated)

            processed += 1
            completed_frames = pass_index * frame_total + processed
            percent = min(99, int(completed_frames * 100 / (frame_total * pass_total)))
            message = "Menghitung baseline…" if write_path is None and pass_total > 1 else "Menganalisis video…"
            _set_video_job(job["job_id"], percent=percent, message=message)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if processed == 0:
        raise ValueError("Video tidak memiliki frame yang dapat dibaca.")
    average_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
    summary = {
        "total": len(seen_ids),
        "inside_region": len(entered_ids),
        "avg_conf": round(average_confidence, 6),
    }
    return last_stats, summary


def _encode_video_h264(source_path, result_path):
    """Re-encode hasil OpenCV menjadi H.264 agar dapat diputar di browser."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-i", str(source_path), "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(result_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not result_path.is_file() or result_path.stat().st_size == 0:
        detail = (completed.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else "ffmpeg tidak menghasilkan file video."
        raise RuntimeError(f"Konversi H.264 gagal: {reason}")


def _process_video_job(job):
    """Proses job dalam worker tunggal dan simpan hasil/progress akhirnya."""
    input_path = job["input_path"]
    temporary_path = OUTPUT_DIR / f"{job['job_id']}_opencv.mp4"
    result_path = OUTPUT_DIR / f"{job['job_id']}_result.mp4"
    try:
        # Siapkan folder hasil di dalam try agar kegagalan disk tercatat di job.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Baca metadata awal untuk menentukan ukuran dan estimasi progress.
        probe = cv2.VideoCapture(str(input_path))
        if not probe.isOpened():
            probe.release()
            raise ValueError("File yang diunggah bukan video valid atau formatnya tidak didukung.")
        source_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_total <= 0:
            # Hitung frame manual hanya untuk container yang tidak memberi metadata.
            frame_total = 0
            while True:
                has_frame, _frame = probe.read()
                if not has_frame:
                    break
                frame_total += 1
        frame_total = max(1, frame_total)
        fps = float(probe.get(cv2.CAP_PROP_FPS))
        probe.release()
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Dimensi video tidak dapat dibaca.")

        # Pertahankan rasio aspek saat menurunkan lebar video menjadi maksimal 960.
        scale = min(1.0, 960 / source_width)
        frame_size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0

        # Baseline berjalan sebagai lintasan terpisah agar tracker tidak berbagi state.
        pass_total = 2 if job["enhance"] != "none" else 1
        baseline = None
        if pass_total == 2:
            _set_video_job(job["job_id"], message="Menyiapkan baseline…")
            _, baseline = _run_video_pass(
                job, job["model_source"], frame_size, fps, frame_total, 0, pass_total
            )

        # Muat instance model baru untuk hasil utama, terpisah dari model global.
        _set_video_job(job["job_id"], message="Menganalisis video…")
        stats, enhanced_summary = _run_video_pass(
            job,
            job["model_source"],
            frame_size,
            fps,
            frame_total,
            pass_total - 1 if pass_total > 1 else 0,
            pass_total,
            write_path=temporary_path,
        )

        # Beri tahu frontend bahwa tracking selesai dan video sedang dikonversi.
        _set_video_job(job["job_id"], percent=99, message="Mengonversi video ke H.264…")
        _encode_video_h264(temporary_path, result_path)
        # Simpan laporan setelah kedua lintasan dan file H.264 berhasil dibuat.
        if baseline is not None:
            _append_enhancement_report(
                job["file_name"],
                "video",
                job["model_name"],
                job["enhance"],
                job["brightness"],
                job["conf"],
                baseline,
                enhanced_summary,
            )
        _set_video_job(
            job["job_id"],
            status="done",
            percent=100,
            result_url=f"/outputs/{result_path.name}",
            stats=stats,
            baseline=baseline,
            message="Pemrosesan video selesai.",
        )
    except Exception as error:
        logger.exception("Job video %s gagal", job["job_id"])
        result_path.unlink(missing_ok=True)
        _set_video_job(
            job["job_id"],
            status="error",
            percent=0,
            result_url=None,
            message=f"Pemrosesan video gagal: {error}",
        )
    finally:
        input_path.unlink(missing_ok=True)
        temporary_path.unlink(missing_ok=True)


def _video_worker_loop():
    """Ambil job satu per satu sehingga hanya satu video diproses bersamaan."""
    while True:
        job = VIDEO_QUEUE.get()
        try:
            _set_video_job(job["job_id"], status="running", message="Job video dimulai.")
            _process_video_job(job)
        finally:
            VIDEO_QUEUE.task_done()


def _ensure_video_worker():
    """Mulai worker daemon sekali saja ketika job video pertama masuk."""
    global VIDEO_WORKER_STARTED
    with VIDEO_WORKER_LOCK:
        if not VIDEO_WORKER_STARTED:
            worker = threading.Thread(target=_video_worker_loop, name="video-worker", daemon=True)
            worker.start()
            VIDEO_WORKER_STARTED = True


@app.post("/api/process_video")
def process_video():
    """Simpan upload dan masukkan pemrosesan video ke antrean background."""
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Pilih file video terlebih dahulu."}), 400

    # Periksa state model dan ambil lokasi bobot untuk instance worker terpisah.
    with MODEL_LOCK:
        active_model_name = ACTIVE_MODEL_NAME
        fallback = ACTIVE_MODEL_IS_FALLBACK
        if ACTIVE_MODEL is None:
            return jsonify({"error": "Model belum tersedia. Upload atau pilih model .pt terlebih dahulu."}), 503
    if fallback:
        model_source = FALLBACK_MODEL_NAME
    else:
        model_source = MODEL_DIR / active_model_name
        if not model_source.is_file():
            return jsonify({"error": "File model aktif tidak ditemukan di folder model/."}), 503

    # Validasi parameter request sama seperti proses deteksi foto.
    with SETTINGS_LOCK:
        default_conf = SETTINGS["conf"]
    conf = _valid_conf(request.form.get("conf", default_conf))
    if conf is None:
        return jsonify({"error": "Confidence harus bernilai antara 0.1 dan 0.9."}), 400
    enhance = request.form.get("enhance", "none").strip().lower()
    if enhance not in {"none", "he", "clahe", "cs", "gamma"}:
        return jsonify({"error": "Metode enhancement tidak dikenal."}), 400
    try:
        brightness = float(request.form.get("brightness", 0))
        if not math.isfinite(brightness) or not -100 <= brightness <= 100:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "Brightness harus bernilai antara -100 dan 100."}), 400

    # Parse region sebagai poligon normalized 0..1 atau biarkan kosong.
    region_value = request.form.get("region", "").strip()
    if region_value:
        try:
            region = json.loads(region_value)
            pixels = region_to_pixels(region, 100, 100)
            if pixels is None:
                raise ValueError
        except (json.JSONDecodeError, TypeError, ValueError):
            return jsonify({"error": "Region harus berupa poligon JSON dengan minimal tiga titik."}), 400
    else:
        region = None

    # Simpan video dengan nama acak dan ekstensi aman ke folder uploads/.
    safe_name = secure_filename(uploaded.filename)
    extension = Path(safe_name).suffix.lower()
    if extension not in {".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv"}:
        return jsonify({"error": "Format video harus MP4, MOV, WebM, M4V, AVI, atau MKV."}), 400
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{extension}"
    try:
        uploaded.save(input_path)
        if input_path.stat().st_size == 0:
            input_path.unlink(missing_ok=True)
            return jsonify({"error": "File video kosong."}), 400
    except OSError as error:
        input_path.unlink(missing_ok=True)
        return jsonify({"error": f"Video gagal disimpan: {error}"}), 500

    # Daftarkan state sebelum antrean agar progress langsung bisa dipolling.
    job = {
        "job_id": job_id,
        "input_path": input_path,
        "model_source": model_source,
        "conf": conf,
        "enhance": enhance,
        "brightness": brightness,
        "region": region,
        "file_name": safe_name,
        "model_name": active_model_name,
        "status": "queued",
        "percent": 0,
        "result_url": None,
        "stats": {"inside_now": 0, "entered_region": 0, "unique_total": 0},
        "baseline": None,
        "message": "Video masuk antrean.",
    }
    # Pastikan model belum diganti/dihapus saat file video diunggah. Model lock
    # mengurutkan pendaftaran job dengan aksi hapus/pilih model.
    with MODEL_LOCK:
        still_active = ACTIVE_MODEL is not None and ACTIVE_MODEL_NAME == active_model_name
        model_exists = fallback or (model_source.is_file() if isinstance(model_source, Path) else False)
        if not still_active or not model_exists:
            input_path.unlink(missing_ok=True)
            return jsonify({"error": "Model berubah saat video diunggah. Pilih model lalu coba lagi."}), 409
        with VIDEO_JOBS_LOCK:
            VIDEO_JOBS[job_id] = job
    _ensure_video_worker()
    VIDEO_QUEUE.put(job)
    return jsonify({"job_id": job_id}), 202


@app.get("/api/progress/<job_id>")
def video_progress(job_id):
    """Kirim status dan statistik terbaru untuk polling frontend."""
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if job is None:
            return jsonify({
                "status": "error",
                "percent": 0,
                "result_url": None,
                "stats": {"inside_now": 0, "entered_region": 0, "unique_total": 0},
                "baseline": None,
                "message": "Pekerjaan video tidak ditemukan.",
            })
        return jsonify({key: job[key] for key in (
            "status", "percent", "result_url", "stats", "baseline", "message"
        )})


@app.get("/api/report")
def get_report():
    """Kirim seluruh baris laporan dari file JSON lokal."""
    with REPORT_LOCK:
        rows = _read_report_rows()
    return jsonify({"rows": rows})


@app.post("/api/report/clear")
def clear_report():
    """Kosongkan log laporan dengan penggantian file yang aman."""
    try:
        with REPORT_LOCK:
            _write_report_rows_atomic([])
    except OSError as error:
        logger.exception("Log laporan gagal dikosongkan")
        return jsonify({"ok": False, "error": f"Laporan gagal dikosongkan: {error}"}), 500
    return jsonify({"ok": True})


def _style_excel_header(worksheet, row_number=1):
    """Beri warna aksen dan teks tebal pada header spreadsheet."""
    from openpyxl.styles import Alignment, Font, PatternFill

    for cell in worksheet[row_number]:
        cell.fill = PatternFill(fill_type="solid", fgColor="1D5FBF")
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    worksheet.row_dimensions[row_number].height = 30


def _fit_excel_columns(worksheet):
    """Sesuaikan lebar berdasarkan isi dengan batas agar tidak terlalu lebar."""
    for column_cells in worksheet.columns:
        letter = column_cells[0].column_letter
        longest = max((len(str(cell.value or "")) for cell in column_cells), default=0)
        worksheet.column_dimensions[letter].width = min(42, max(12, longest + 2))


def _report_number(row, key):
    """Konversi angka dari JSON untuk ditaruh ke sel Excel."""
    value = row.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _make_report_workbook(rows):
    """Buat workbook laporan lengkap dan ringkasan rata-rata per enhancement."""
    from collections import defaultdict
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    headers = [
        "Waktu", "Nama file", "Jenis", "Model", "Metode enhancement",
        "Brightness", "Confidence", "Total sebelum", "Total sesudah",
        "Selisih total", "Region sebelum", "Region sesudah", "Selisih region",
        "Avg conf sebelum", "Avg conf sesudah", "Selisih avg conf",
    ]
    fields = [
        "waktu", "nama_file", "jenis", "model", "metode_enhancement",
        "brightness", "conf", "total_sebelum", "total_sesudah", "selisih_total",
        "region_sebelum", "region_sesudah", "selisih_region", "avgconf_sebelum",
        "avgconf_sesudah", "selisih_avgconf",
    ]
    numeric_fields = fields[5:]
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Laporan"
    sheet.append(headers)

    for row in rows:
        values = [row.get(field) for field in fields]
        for index in range(5, len(values)):
            values[index] = _report_number(row, fields[index])
        sheet.append(values)
    _style_excel_header(sheet)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:P{max(1, len(rows) + 1)}"

    # Hitung rata-rata di Python agar tetap langsung terbaca oleh pembaca XLSX.
    average_row = ["Rata-rata"] + [None] * (len(fields) - 1)
    for column_index, field in enumerate(numeric_fields, start=6):
        numbers = [number for row in rows if (number := _report_number(row, field)) is not None]
        average_row[column_index - 1] = sum(numbers) / len(numbers) if numbers else None
    sheet.append(average_row)
    footer_index = sheet.max_row
    for cell in sheet[footer_index]:
        cell.fill = PatternFill(fill_type="solid", fgColor="E7EEF9")
        cell.font = Font(bold=True, color="182131")

    for row_index in range(2, sheet.max_row + 1):
        for column_index in (6, 7, 14, 15, 16):
            sheet.cell(row_index, column_index).number_format = "0.00"
        sheet.cell(row_index, 16).number_format = "+0.00;-0.00;0.00"
    _fit_excel_columns(sheet)

    # Kelompokkan tiga metrik selisih untuk ringkasan masing-masing metode.
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("metode_enhancement") or "none"].append(row)
    summary = workbook.create_sheet("Ringkasan")
    summary.append([
        "Metode enhancement", "Jumlah laporan", "Rata-rata selisih total",
        "Rata-rata selisih region", "Rata-rata selisih avg conf",
    ])
    for method in sorted(groups):
        method_rows = groups[method]
        averages = []
        for field in ("selisih_total", "selisih_region", "selisih_avgconf"):
            numbers = [number for row in method_rows if (number := _report_number(row, field)) is not None]
            averages.append(sum(numbers) / len(numbers) if numbers else None)
        summary.append([method, len(method_rows), *averages])
    _style_excel_header(summary)
    summary.freeze_panes = "A2"
    for row_index in range(2, summary.max_row + 1):
        for column_index in range(3, 6):
            summary.cell(row_index, column_index).number_format = "0.00"
    _fit_excel_columns(summary)

    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)
    return output


@app.get("/api/export_excel")
def export_excel():
    """Ekspor log laporan menjadi workbook dengan lembar data dan ringkasan."""
    with REPORT_LOCK:
        rows = _read_report_rows()
    try:
        output = _make_report_workbook(rows)
    except Exception as error:
        logger.exception("Workbook laporan gagal dibuat")
        return jsonify({"error": f"File Excel gagal dibuat: {error}"}), 500
    return send_file(
        output,
        as_attachment=True,
        download_name="laporan-enhancement.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.post("/api/settings")
def update_settings():
    """Validasi dan simpan confidence dengan penggantian file atomik."""
    data = request.get_json(silent=True) or {}
    conf = _valid_conf(data.get("conf"))
    if conf is None:
        return jsonify({"ok": False}), 400

    try:
        with SETTINGS_LOCK:
            updated = dict(SETTINGS)
            updated["conf"] = conf
            _write_settings_atomic(updated)
            SETTINGS.clear()
            SETTINGS.update(updated)
    except OSError as error:
        logger.error("Confidence gagal disimpan: %s", error)
        return jsonify({"ok": False}), 500
    return jsonify({"ok": True})


@app.get("/outputs/<path:filename>")
def get_output(filename):
    """Sajikan file hasil hanya dari folder outputs/."""
    return send_from_directory(OUTPUT_DIR, filename)


# Baca pengaturan dan muat model pilihan saat modul/server mulai.
initialize_app_state()


if __name__ == "__main__":
    # Matikan reloader agar model tidak dimuat dua kali saat development.
    app.run(debug=True, use_reloader=False)

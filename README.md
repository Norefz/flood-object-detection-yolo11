# Flood Object Detection YOLO11

Aplikasi web Flask untuk mendeteksi objek pada kondisi banjir memakai model YOLO11. Aplikasi mendukung deteksi foto dan video, enhancement gambar, penghitungan dalam satu region poligon, serta laporan perbandingan sebelum dan sesudah enhancement.

## Teknologi

- Python
- Flask
- HTML, CSS, dan JavaScript biasa
- Ultralytics YOLO11
- OpenCV
- openpyxl
- NumPy
- imageio-ffmpeg untuk encoding video H.264 saat FFmpeg sistem tidak tersedia

## Instalasi

Pastikan Python dan `venv` tersedia, lalu buat dan aktifkan virtual environment dari folder proyek.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Menjalankan aplikasi

Aktifkan virtual environment, lalu jalankan:

```bash
python app.py
```

Buka <http://127.0.0.1:5000> di browser. Saat server mulai, file di `uploads/` dan `outputs/` yang lebih lama dari 24 jam akan dibersihkan.

## Upload model

1. Jalankan aplikasi dan buka tab **Foto** atau **Pengaturan**.
2. Tekan **Upload model .pt** atau **Upload model baru**, lalu pilih bobot YOLO berformat `.pt`.
3. Model yang berhasil dimuat disimpan di `model/` dan langsung menjadi model aktif. Pilihan model aktif disimpan di `settings.json`.
4. Batas ukuran file model adalah 200 MB. Batas keseluruhan request upload aplikasi adalah 500 MB.

Gunakan bobot yang dilatih untuk kelas objek banjir yang diinginkan. Aplikasi tidak mengunduh model fallback secara otomatis. Fallback hanya dicoba bila server dijalankan dengan `USE_FALLBACK=1`.

## Struktur folder

```text
app.py                  Route Flask dan pemrosesan foto/video
settings.json           Model aktif dan confidence default
requirements.txt        Dependensi Python
model/                  Bobot model YOLO .pt
uploads/                Video yang sedang diproses
outputs/                Hasil foto dan video
reports/                report_log.json dan laporan deteksi
utils/enhance.py        Image enhancement
utils/counter.py        Region dan penghitungan deteksi
templates/index.html    Struktur halaman web
static/style.css        Styling halaman
static/app.js           Interaksi frontend dan pemanggilan API
```

## Checklist uji manual

- [ ] **Model:** upload bobot `.pt` yang valid; pastikan tampil sebagai model aktif. Coba pilih model lain dan pastikan model aktif tidak bisa dihapus.
- [ ] **Foto:** deteksi foto valid; pastikan foto asli, enhanced, dan hasil tampil. Coba file non-gambar dan pastikan pesan error muncul.
- [ ] **Video:** proses video valid sampai progress selesai; putar dan unduh hasil `.mp4`. Coba file non-video dan pastikan error ditampilkan.
- [ ] **Region:** proses tanpa region; gambar poligon dengan minimal tiga titik dan cek angka di dalam region. Coba menyelesaikan region dengan kurang dari tiga titik.
- [ ] **Enhancement:** uji `none`, `he`, `clahe`, `cs`, dan `gamma`, dengan beberapa nilai brightness; pastikan perbandingan sebelum/sesudah muncul bila baseline tersedia.
- [ ] **Laporan:** buka tab Laporan setelah deteksi dengan enhancement; pastikan angka sebelum/sesudah dan selisih tercatat. Tekan **Kosongkan** dan pastikan daftar menjadi kosong.
- [ ] **Export Excel:** ekspor laporan; buka file dan periksa sheet **Laporan**, baris rata-rata, serta sheet **Ringkasan**.
- [ ] **Upload besar:** pastikan request di atas 500 MB ditolak dengan pesan batas ukuran yang jelas.

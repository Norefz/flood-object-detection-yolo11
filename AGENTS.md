# Konteks proyek

Project ini adalah aplikasi deteksi kendaraan banjir memakai YOLO11 dengan satu kelas, yaitu `vehicle`, dan GUI web berbasis Flask. Fitur pilihan adalah **COUNTING**, bukan tracking, dengan satu region poligon. ByteTrack hanya digunakan di belakang agar kendaraan tidak dihitung ganda saat diproses dari video.

Fitur aplikasi yang direncanakan:

- Deteksi foto dan video.
- Upload dan pemilihan model `.pt` yang disimpan di `model/`.
- Model aktif disimpan di `settings.json`; aplikasi tidak memakai database.
- Image enhancement: `none`, `he`, `clahe`, `cs`, serta `gamma` dan `brightness`.
- Region counter dan laporan Excel sebelum/sesudah enhancement.

## Aturan kerja

- Kerjakan hanya bagian yang diminta pengguna; jangan ubah file lain.
- Beri komentar singkat dalam bahasa Indonesia.
- Gunakan HTML, CSS, dan JavaScript biasa tanpa framework.

## Kontrak API

Frontend dan backend harus mengikuti kontrak berikut persis:

- `GET /api/models` -> `{models:[{name,size_mb,uploaded}], active:"x.pt"|null, fallback:bool}`
- `POST /api/upload_model` (form: `file` `.pt`) -> `{ok, active}` atau `{ok:false,error}`
- `POST /api/select_model` (JSON: `{name}`) -> `{ok, active}`
- `POST /api/delete_model` (JSON: `{name}`) -> `{ok}`
- `GET /api/settings` -> `{conf}`
- `POST /api/settings` (JSON: `{conf}`) -> `{ok}`
- `POST /api/detect_image` (form: `file`, `conf`, `enhance`, `brightness`, `region`) -> `{original_url, enhanced_url, result_url, total, inside_region, avg_conf, baseline:{total,inside_region,avg_conf}|null, model}`
- `POST /api/process_video` (form yang sama seperti `detect_image`, dengan file video) -> `{job_id}`
- `GET /api/progress/<job_id>` -> `{status:"queued|running|done|error", percent, result_url, stats:{inside_now, entered_region, unique_total}, baseline:{...}|null, message}`
- `GET /api/report` -> `{rows:[...]}`
- `POST /api/report/clear` -> `{ok}`
- `GET /api/export_excel` -> file `.xlsx`
- `GET /outputs/<filename>` -> file hasil

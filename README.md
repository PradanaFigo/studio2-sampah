# Studio 2 — Klasifikasi Sampah · Kelompok <NOMOR/NAMA KELOMPOK>

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  |              |     |                  |
| 2  |              |     |                  |
| 3  |              |     |                  |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: *(isi setelah eksperimen, mis. ConvNeXt-Tiny pretrained ImageNet, head baru 5 kelas)*
- **Strategi pelatihan**: transfer learning dengan tiga strategi yang dibandingkan — `linear_probe`
  (backbone dibekukan), `gradual` (linear probing lalu semua layer dibuka, LR backbone ×0,1), dan
  `full` (semua layer dilatih sejak awal). AdamW, cosine scheduler, label smoothing 0,1, mixed precision (AMP).
  Checkpoint dipilih berdasarkan **Macro-F1** pada split validasi lokal 15%.
- **Data tambahan**: tidak ada *(ubah bila memakai data luar dan sebutkan sumber + jumlahnya)*
- **Ukuran input & preprocessing**: *(mis. 224×224; train: RandomResizedCrop(scale 0,5–1), flip, ColorJitter; validasi: Resize → CenterCrop; normalisasi ImageNet)*

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | | | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

### Perbandingan backbone dan strategi (dari `runs/results.csv`)

| Backbone | Strategi | Param total | Param dilatih | Macro-F1 lokal | Waktu (dtk) |
|----------|----------|-------------|---------------|----------------|-------------|
|          |          |             |               |                |             |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

-

## Cara Reproduksi

```bash
pip install -r requirements.txt

# 1. Latih satu konfigurasi (bobot pretrained ImageNet diunduh otomatis oleh torchvision)
python train.py --data ./train --backbone convnext_tiny --strategy gradual --epochs 12 --export

# 2. Uji submission pada split validasi lokal (dibuat otomatis oleh train.py)
python evaluate.py --submission ./submission --data ./my_val
```

`train.py` membuat `my_train/` dan `my_val/` (15% per kelas, seed tetap) dari folder `train/`.
Dataset **tidak** disertakan di repo ini sesuai aturan kompetisi.

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform() (membaca config.json)
  model.pth         # state_dict model terbaik
  config.json       # backbone + ukuran input, ditulis otomatis oleh train.py
evaluate.py         # salinan skrip evaluasi resmi
train.py            # training: pilih backbone dan strategi fine-tuning
colab_train.ipynb   # notebook Colab untuk menjalankan eksperimen
runs/results.csv    # tabel hasil semua eksperimen
requirements.txt
```

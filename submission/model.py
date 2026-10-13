"""Submission Studio 2 — Klasifikasi Sampah (IF25-40401).

Arsitektur dibangun dari torchvision (weights=None), lalu head diganti menjadi 5 kelas.
Backbone dan ukuran input dibaca dari `config.json` di folder yang sama; berkas itu
ditulis otomatis oleh train.py bersama model.pth. Jadi model.py TIDAK perlu diedit
setiap kali berganti backbone.

Aturan kontrak (lihat halaman kompetisi):
  - get_model() hanya MEMBANGUN arsitektur: tanpa unduh bobot, tanpa memuat model.pth.
  - get_transform() sama dengan transform validasi saat training.
  - Output: logits [batch, 5]; urutan kelas alfabetis:
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
"""
import json
from pathlib import Path

import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Nilai bawaan bila config.json tidak ditemukan.
DEFAULT_CONFIG = {"backbone": "resnet18", "img_size": 224}


def load_config():
    path = Path(__file__).resolve().parent / "config.json"
    if path.exists():
        return {**DEFAULT_CONFIG, **json.loads(path.read_text())}
    return dict(DEFAULT_CONFIG)


def head_module(model, backbone):
    """Layer Linear terakhir (head klasifikasi) untuk tiap keluarga arsitektur."""
    if backbone.startswith("resnet"):
        return model.fc
    if backbone.startswith("mobilenet_v3"):
        return model.classifier[3]
    if backbone.startswith("efficientnet"):
        return model.classifier[1]
    if backbone.startswith("convnext"):
        return model.classifier[2]
    raise ValueError(f"Backbone belum didukung: {backbone}")


def replace_head(model, backbone, n_classes=NUM_CLASSES):
    """Ganti head 1000 kelas ImageNet dengan Linear(n_classes) (slide 41 / Notebook 05 bagian 6b)."""
    old = head_module(model, backbone)
    new = nn.Linear(old.in_features, n_classes)
    if backbone.startswith("resnet"):
        model.fc = new
    elif backbone.startswith("mobilenet_v3"):
        model.classifier[3] = new
    elif backbone.startswith("efficientnet"):
        model.classifier[1] = new
    elif backbone.startswith("convnext"):
        model.classifier[2] = new
    return model


def build(backbone, pretrained=False):
    """pretrained=True hanya dipakai train.py; get_model() selalu pretrained=False."""
    model = models.get_model(backbone, weights="DEFAULT" if pretrained else None)
    return replace_head(model, backbone)


def get_model():
    return build(load_config()["backbone"], pretrained=False)


def eval_transform(img_size):
    # Resize sisi pendek lebih besar sedikit lalu center crop (rasio 224 -> 256 seperti baseline).
    return transforms.Compose([
        transforms.Resize(round(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_transform():
    return eval_transform(load_config()["img_size"])

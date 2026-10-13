"""Training Studio 2 — Klasifikasi Sampah (IF25-40401).

Satu skrip untuk membandingkan backbone dan strategi fine-tuning secara sistematis.

Backbone (torchvision) : resnet18, resnet50, mobilenet_v3_large, efficientnet_b0, convnext_tiny
Strategi fine-tuning   :
    linear_probe : backbone dibekukan, hanya head baru yang dilatih
    gradual      : fase 1 linear probing (--warmup-epochs), fase 2 semua layer dibuka,
                   backbone memakai LR = lr * --backbone-lr-mult (head tetap lr)
    full         : semua layer dilatih dari epoch 1 dengan LR yang sama

Contoh:
    python train.py --data ./train --backbone convnext_tiny --strategy gradual --epochs 12
    python train.py --data ./train --backbone resnet18 --strategy full --export

Keluaran per run (folder runs/<tag>/): model.pth, config.json, metrics.json, dan satu baris
ditambahkan ke runs/results.csv (tabel perbandingan untuk README).
`--export` menyalin run tersebut ke submission/ (model.pth + config.json).
"""
import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from evaluate import CLASSES, metrics
from submission.model import MEAN, STD, build, eval_transform, head_module

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_local_split(src, dst, frac, seed):
    """Salin frac gambar per kelas ke my_val/ (sekali saja), sisanya ke my_train/."""
    dst_tr, dst_va = Path(dst) / "my_train", Path(dst) / "my_val"
    if dst_va.exists():
        return dst_tr, dst_va
    rng = random.Random(seed)
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in EXTS)
        rng.shuffle(files)
        n_va = int(len(files) * frac)
        for i, f in enumerate(files):
            out = (dst_va if i < n_va else dst_tr) / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
    return dst_tr, dst_va


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def set_frozen(model, backbone, frozen):
    """frozen=True: hanya head yang requires_grad (linear probing)."""
    head = head_module(model, backbone)
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = (not frozen) or (id(p) in head_ids)


def make_optimizer(model, backbone, lr, backbone_lr_mult, wd, separate_groups):
    head = head_module(model, backbone)
    head_ids = {id(p) for p in head.parameters()}
    head_p = [p for p in model.parameters() if id(p) in head_ids]
    body_p = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    groups = [{"params": head_p, "lr": lr}]
    if body_p:
        groups.append({"params": body_p, "lr": lr * (backbone_lr_mult if separate_groups else 1.0)})
    return torch.optim.AdamW(groups, weight_decay=wd)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()
    return metrics(y_true, y_pred)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder train/ dari dataset (berisi 5 folder kelas)")
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet50", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny"])
    ap.add_argument("--strategy", default="gradual", choices=["linear_probe", "gradual", "full"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--warmup-epochs", type=int, default=2, help="epoch linear probing pada strategi gradual")
    ap.add_argument("--lr", type=float, default=1e-3, help="LR head (full: LR semua layer)")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1, help="pengali LR backbone (gradual)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--split-dir", default=".", help="tempat my_train/ dan my_val/ dibuat")
    ap.add_argument("--no-pretrained", action="store_true", help="tanpa bobot ImageNet (untuk uji kode/ablasi)")
    ap.add_argument("--full-data", action="store_true",
                    help="latih di train+my_val (untuk model final); simpan epoch terakhir, bukan yang terbaik")
    ap.add_argument("--tag", default=None, help="nama run; bawaan: <backbone>_<strategy>_<img>_s<seed>")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--export", action="store_true", help="salin hasil ke submission/")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device == "cuda"
    tag = a.tag or f"{a.backbone}_{a.strategy}_{a.img_size}_s{a.seed}"
    out_dir = Path(a.runs_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_dir, va_dir = make_local_split(a.data, a.split_dir, a.val_frac, a.seed)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(a.img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tr_ds = datasets.ImageFolder(tr_dir, train_tf)
    va_ds = datasets.ImageFolder(va_dir, eval_transform(a.img_size))
    assert tr_ds.classes == CLASSES, f"urutan kelas tidak cocok: {tr_ds.classes}"
    if a.full_data:
        tr_ds = ConcatDataset([tr_ds, datasets.ImageFolder(va_dir, train_tf)])
    print(f"[{tag}] device={device} amp={use_amp} train={len(tr_ds)} val={len(va_ds)}")
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True, num_workers=a.workers,
                       pin_memory=use_amp, drop_last=True)
    va_dl = DataLoader(va_ds, a.batch_size, num_workers=a.workers)

    model = build(a.backbone, pretrained=not a.no_pretrained).to(device)
    n_total = count_params(model)

    # Fase awal: linear_probe & gradual membekukan backbone; full langsung membuka semua layer.
    frozen = a.strategy in ("linear_probe", "gradual")
    set_frozen(model, a.backbone, frozen)
    opt = make_optimizer(model, a.backbone, a.lr, a.backbone_lr_mult, a.weight_decay, separate_groups=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_trained = count_params(model, trainable_only=True)

    best_f1, best_ep, best_m, t0 = -1.0, 0, None, time.time()
    for ep in range(1, a.epochs + 1):
        # Strategi gradual: buka seluruh backbone setelah warmup, LR backbone lebih kecil.
        if a.strategy == "gradual" and frozen and ep == a.warmup_epochs + 1:
            frozen = False
            set_frozen(model, a.backbone, False)
            opt = make_optimizer(model, a.backbone, a.lr, a.backbone_lr_mult, a.weight_decay, separate_groups=True)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs - a.warmup_epochs)
            n_trained = count_params(model, trainable_only=True)
            print(f"  -> backbone dibuka (LR backbone x{a.backbone_lr_mult})")

        model.train()
        if frozen:
            model.eval()  # linear probing murni: BatchNorm/Dropout backbone tidak berubah
        run_loss = 0.0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
        sched.step()

        m = evaluate_loader(model, va_dl, device)
        print(f"  epoch {ep:2d}/{a.epochs}  loss={run_loss / max(1, len(tr_dl)):.4f}  "
              f"val_macroF1={m['macro_f1']:.4f}  val_acc={m['accuracy']:.4f}")
        if a.full_data:  # tanpa seleksi: simpan epoch terakhir
            best_f1, best_ep, best_m = m["macro_f1"], ep, m
            torch.save(model.state_dict(), out_dir / "model.pth")
        elif m["macro_f1"] > best_f1:
            best_f1, best_ep, best_m = m["macro_f1"], ep, m
            torch.save(model.state_dict(), out_dir / "model.pth")

    secs = round(time.time() - t0)
    (out_dir / "config.json").write_text(json.dumps({"backbone": a.backbone, "img_size": a.img_size}))
    (out_dir / "metrics.json").write_text(json.dumps({**best_m, "best_epoch": best_ep, "args": vars(a)}, indent=2))

    row = {"tag": tag, "backbone": a.backbone, "strategy": a.strategy, "img_size": a.img_size,
           "epochs": a.epochs, "lr": a.lr, "seed": a.seed, "pretrained": not a.no_pretrained,
           "full_data": a.full_data, "params_total": n_total,
           "params_trained_final": n_trained, "best_epoch": best_ep,
           "macro_f1": round(best_m["macro_f1"], 4), "accuracy": round(best_m["accuracy"], 4),
           **{f"f1_{c}": round(f, 4) for c, f in best_m["per_class_f1"].items()}, "seconds": secs}
    csv_path = Path(a.runs_dir) / "results.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{tag}] selesai: best macro-F1={best_f1:.4f} (epoch {best_ep}), {secs}s -> {out_dir}")
    if a.full_data:
        print("  CATATAN: --full-data memakai my_val sebagai data latih, jadi skor di atas optimistis.")

    if a.export:
        Path("submission").mkdir(exist_ok=True)
        shutil.copy2(out_dir / "model.pth", "submission/model.pth")
        shutil.copy2(out_dir / "config.json", "submission/config.json")
        print("  diekspor ke submission/model.pth + submission/config.json\n"
              "  uji: python evaluate.py --submission ./submission --data ./my_val")


if __name__ == "__main__":
    main()

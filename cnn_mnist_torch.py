#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_mnist_torch.py —— PyTorch 卷积神经网络训练真实 MNIST（主交付程序）

特点
    * 经典两层卷积 + 批归一化 + Dropout 的 CNN
    * 自动选择设备（CUDA / Apple MPS / CPU）
    * 训练集随机增强（小幅平移+旋转），测试集不做增强
    * 支持 CPU/GPU、断点保存、学习率调度、早停（可选）
    * 无需 matplotlib：训练曲线/预测样例/混淆矩阵用 Pillow 直接画出来
    * 自动定位 MNIST：优先 ./data/MNIST/raw/，也兼容 torchvision 自动下载
    * 兼容有无 torchvision：没有 torchvision 时会读 IDX 原始文件

安装依赖
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # 仅 CPU
    pip install torch torchvision                                                     # 有 NVIDIA 显卡

快速开始
    python cnn_mnist_torch.py                       # 训练 5 个 epoch
    python cnn_mnist_torch.py --epochs 10 --device cuda
    python cnn_mnist_torch.py --eval-only --ckpt outputs/mnist_cnn.pt
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import datasets, transforms
except ImportError:  # 没装 torchvision 时走内置 IDX 读取器
    datasets = None
    transforms = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageFont = None

# MNIST 全局均值/标准差（业界通用值），供标准化与可视化复用
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


# ------------------------------------------------------------------------------
# 模型
# ------------------------------------------------------------------------------
class MNISTCNN(nn.Module):
    """
    Conv(1->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> MaxPool -> Dropout
    Conv(32->64) -> BN -> ReLU -> Conv(64->64) -> BN -> ReLU -> MaxPool -> Dropout
    AdaptiveAvgPool -> Flatten -> FC(128) -> ReLU -> Dropout -> FC(10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
        )
        # 自适应池化，输入尺寸变化时也不会崩
        self.gap = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gap(self.features(x)))


# ------------------------------------------------------------------------------
# 数据集：优先 torchvision，其次直接读 IDX 原始文件
# ------------------------------------------------------------------------------
MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
IDX_FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}


class ArrayDataset(Dataset):
    """把 (N,1,28,28) float32 的数组包成 Dataset（label 可以是 list 或 ndarray）。"""

    def __init__(self, images: np.ndarray, labels, transform=None):
        self.images = torch.from_numpy(np.asarray(images)).float()
        self.labels = torch.from_numpy(np.asarray(labels)).long()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img, label = self.images[i], self.labels[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def augment_image(img: torch.Tensor) -> torch.Tensor:
    """
    随机小角度旋转 + 随机平移（affine_grid + grid_sample）。
    同时兼容 (C,H,W) 和 (B,C,H,W)：grid_sample 在 torch 2.x 对维度校验很严格，
    所以这里显式保证网格与输入都是 4D。
    """
    squeeze = img.dim() == 3
    x = img.unsqueeze(0) if squeeze else img
    if x.dim() != 4:
        raise ValueError(f"augment_image 需要 3D 或 4D 输入，收到 {tuple(img.shape)}")

    angle = float(torch.empty(1).uniform_(-12, 12)) * 3.14159265 / 180
    max_shift = 0.10
    tx = float(torch.empty(1).uniform_(-max_shift, max_shift))
    ty = float(torch.empty(1).uniform_(-max_shift, max_shift))

    cos, sin = float(torch.cos(torch.tensor(angle))), float(torch.sin(torch.tensor(angle)))
    theta = torch.tensor([[[cos, -sin, tx], [sin, cos, ty]]],
                         dtype=x.dtype, device=x.device).expand(x.size(0), 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, align_corners=False)
    return out.squeeze(0) if squeeze else out


def to_display(img: torch.Tensor, mean: float = MNIST_MEAN, std: float = MNIST_STD) -> torch.Tensor:
    """反标准化，把张量还原成 0~1 的灰度图，便于可视化。"""
    return (img * std + mean).clamp(0.0, 1.0)


def find_idx_dir(root: str = "data") -> str | None:
    """在几个常见位置找 MNIST IDX 文件。"""
    candidates = [
        os.path.join(root, "MNIST", "raw"),
        root,
        os.path.join("data", "MNIST", "raw"),
    ]
    for d in candidates:
        if all(os.path.exists(os.path.join(d, f)) for f in IDX_FILES.values()):
            return d
    return None


def _read_idx(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * (ndim - 1), f.read(4 * (ndim - 1)))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


def load_idx_data(directory: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xtr = _read_idx(os.path.join(directory, IDX_FILES["train_images"])).astype(np.float32) / 255.0
    ytr = _read_idx(os.path.join(directory, IDX_FILES["train_labels"])).astype(np.int64)
    xte = _read_idx(os.path.join(directory, IDX_FILES["test_images"])).astype(np.float32) / 255.0
    yte = _read_idx(os.path.join(directory, IDX_FILES["test_labels"])).astype(np.int64)
    return xtr[:, None], ytr, xte[:, None], yte


def make_train_transform(augment: bool = True):
    """
    标准数据管线：ToTensor -> (随机仿射增强) -> Normalize。
    放在 transform 里而不是训练循环里，配合 DataLoader 的多进程更高效。
    """
    steps: list = [transforms.ToTensor()]
    if augment:
        steps.append(transforms.Lambda(augment_image))
    steps.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    return transforms.Compose(steps)


def make_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def load_mnist(root: str = "data", download: bool = True, augment: bool = True):
    """
    返回 (train_ds, test_ds, source)。返回的数据集**已经带好 transform**，
    即 DataLoader 直接吐出来的就是标准化后的张量；augment 控制训练集是否做随机增强。
    """
    idx_dir = find_idx_dir(root)

    try:
        from torchvision import datasets, transforms  # noqa: F401

        train = datasets.MNIST(root, train=True, download=download and idx_dir is None,
                               transform=make_train_transform(augment))
        test = datasets.MNIST(root, train=False, download=download and idx_dir is None,
                              transform=make_eval_transform())
        return train, test, "torchvision"
    except ImportError:
        print("[info] 未安装 torchvision，改用内置 IDX 读取器", file=sys.stderr)
    except Exception as exc:
        print(f"[info] torchvision 加载失败({exc})，改用内置 IDX 读取器", file=sys.stderr)

    if idx_dir is None:
        raise RuntimeError(
            "找不到 MNIST 数据。请任选一种方式：\n"
            f"  1) pip install torchvision 后重跑（会自动下载到 {root}/MNIST/raw）\n"
            f"  2) 手动下载 4 个 IDX 文件到 {root}/MNIST/raw/，镜像: {MNIST_MIRRORS[0]}\n"
            "     train-images-idx3-ubyte, train-labels-idx1-ubyte,\n"
            "     t10k-images-idx3-ubyte,  t10k-labels-idx1-ubyte\n"
            "  3) 先运行 python cnn_numpy.py（无需任何下载的离线版本）"
        )
    xtr, ytr, xte, yte = load_idx_data(idx_dir)
    tr = ArrayDataset(xtr, ytr, transform=make_train_transform(augment))
    te = ArrayDataset(xte, yte, transform=make_eval_transform())
    return tr, te, f"idx:{idx_dir}"


# ------------------------------------------------------------------------------
# 训练 / 评估
# ------------------------------------------------------------------------------
@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: str, device: torch.device) -> dict:
    """
    读取我们自己存的 checkpoint。
    PyTorch 2.6 起 torch.load 默认 weights_only=True，而我们的 checkpoint 里
    除了权重还存了 args 字典，所以必须显式 weights_only=False（老版本没有这个参数）。
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 没有 weights_only 参数
        return torch.load(path, map_location=device)


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None,
              label_smoothing: float = 0.0) -> tuple[float, float]:
    """跑一轮。数据已由 transform 完成标准化，这里不再重复处理。"""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing if train else 0.0)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            seen += len(y)
    return total_loss / seen, correct / seen


def evaluate_full(model: nn.Module, loader: DataLoader,
                  device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    """完整评估，返回 (loss, 预测, 真实标签)。"""
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            losses.append(float(F.cross_entropy(logits, y.to(device)).item()) * len(y))
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(y.numpy())
    n = sum(len(t) for t in targets)
    return float(np.sum(losses) / n), np.concatenate(preds), np.concatenate(targets)


def train(args) -> History:
    device = pick_device(args.device)
    print(f"设备: {device} | torch {torch.__version__}")

    train_ds, test_ds, source = load_mnist(args.data, download=not args.no_download,
                                           augment=not args.no_augment)
    print(f"数据来源: {source} | 训练样本 {len(train_ds)} | 测试样本 {len(test_ds)}")
    print(f"增强: {'关' if args.no_augment else '开'} | 标签平滑: {args.label_smoothing}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=args.workers, pin_memory=pin)

    model = MNISTCNN(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    hist = History()
    best_acc = 0.0
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "mnist_cnn.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer,
                                    label_smoothing=args.label_smoothing)
        val_loss, val_acc = run_epoch(model, test_loader, device, None)
        scheduler.step()

        hist.train_loss.append(tr_loss)
        hist.train_acc.append(tr_acc)
        hist.val_loss.append(val_loss)
        hist.val_acc.append(val_acc)
        hist.lr.append(optimizer.param_groups[0]["lr"])
        hist.seconds.append(time.time() - t0)

        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "val_acc": best_acc,
                        "epoch": epoch}, ckpt_path)
            flag = "  *saved*"
        print(f"epoch {epoch:2d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f} "
              f"| lr {hist.lr[-1]:.2e} | {hist.seconds[-1]:.1f}s{flag}")

        if args.patience and epoch - int(np.argmax(hist.val_acc)) - 1 >= args.patience:
            print(f"验证准确率 {args.patience} 个 epoch 没有提升，提前停止")
            break

    print(f"\n最佳验证准确率: {best_acc:.4f} | 权重: {ckpt_path}")

    # 用最佳权重做最终评估与可视化
    if os.path.exists(ckpt_path):
        model.load_state_dict(load_checkpoint(ckpt_path, device)["model"])
    loss, pred, target = evaluate_full(model, test_loader, device)
    acc = float((pred == target).mean())
    print(f"测试集: loss {loss:.4f} | 准确率 {acc:.4f}")
    print("各类别准确率:")
    for c in range(10):
        m = target == c
        print(f"  {c}: {float((pred[m] == c).mean()):.4f}  (n={int(m.sum())})")

    save_history_plot(hist, os.path.join(args.outdir, "torch_history.png"))
    save_prediction_grid(model, test_loader, device, os.path.join(args.outdir, "torch_predictions.png"))
    save_confusion(confusion_matrix(pred, target), os.path.join(args.outdir, "torch_confusion.png"))
    save_error_examples(model, test_loader, device, os.path.join(args.outdir, "torch_errors.png"))
    print(f"结果图片已写入 {args.outdir}/")
    return hist


def confusion_matrix(pred: np.ndarray, y: np.ndarray, classes: int = 10) -> np.ndarray:
    cm = np.zeros((classes, classes), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


# ------------------------------------------------------------------------------
# 可视化（Pillow，免 matplotlib）
# ------------------------------------------------------------------------------
def _font(size: int = 14):
    if ImageFont is None:
        return None
    for path in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def save_history_plot(hist: History, path: str) -> None:
    if Image is None:
        return
    w, h, m = 760, 320, 44
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f = _font(14)
    d.rectangle([m, m, w - m, h - m], outline="#cccccc")

    def poly(vals, color, vmax, vmin=0.0):
        pts = [(m + i * (w - 2 * m) / max(1, len(vals) - 1),
                h - m - (v - vmin) / max(1e-9, vmax - vmin) * (h - 2 * m)) for i, v in enumerate(vals)]
        if len(pts) > 1:
            d.line(pts, fill=color, width=2)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)

    vmax = max(hist.train_loss + hist.val_loss) * 1.1
    poly(hist.train_loss, "#1f6fd0", vmax)
    poly(hist.val_loss, "#d03a1f", vmax)
    d.text((m, 14), "loss  blue=train red=val", fill="#333333", font=f)
    d.text((m, h - 24), "val acc: " + " ".join(f"{a:.3f}" for a in hist.val_acc), fill="#333333", font=f)
    img.save(path)


def _tile(tensor: torch.Tensor, size: int) -> "Image.Image":
    """tensor 是标准化后的 (1,28,28)，先反标准化再转成图片。"""
    arr = (to_display(tensor.squeeze().cpu()) * 255).numpy().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.NEAREST).convert("RGB")


def save_prediction_grid(model: nn.Module, loader: DataLoader, device: torch.device, path: str,
                         cols: int = 8, rows: int = 4) -> None:
    if Image is None:
        return
    model.eval()
    xs, ys = next(iter(loader))
    with torch.no_grad():
        pred = model(xs.to(device)).argmax(1).cpu()
    n = min(cols * rows, len(xs))
    cell, pad = 84, 6
    img = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + pad + 18) + pad), "white")
    d = ImageDraw.Draw(img)
    f = _font(15)
    for i in range(n):
        r, c = divmod(i, cols)
        x0, y0 = pad + c * (cell + pad), pad + r * (cell + pad + 18)
        img.paste(_tile(xs[i], cell), (x0, y0))
        ok = int(pred[i]) == int(ys[i])
        d.rectangle([x0, y0, x0 + cell, y0 + cell], outline="#22aa44" if ok else "#cc2222", width=2)
        d.text((x0, y0 + cell + 2), f"p={int(pred[i])} t={int(ys[i])}", fill="#222222", font=f)
    img.save(path)


def save_error_examples(model: nn.Module, loader: DataLoader, device: torch.device, path: str,
                        limit: int = 32) -> None:
    """挑出判错的样本，看看模型错在哪。"""
    if Image is None:
        return
    model.eval()
    wrong_x, wrong_p, wrong_t = [], [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            bad = pred != y
            wrong_x.append(x[bad])
            wrong_p.append(pred[bad])
            wrong_t.append(y[bad])
            if sum(len(t) for t in wrong_t) >= limit:
                break
    if not wrong_t:
        return
    xs = torch.cat(wrong_x)[:limit]
    ps = torch.cat(wrong_p)[:limit]
    ts = torch.cat(wrong_t)[:limit]

    cols = 8
    rows = (len(xs) + cols - 1) // cols
    cell, pad = 84, 6
    img = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + pad + 18) + pad), "white")
    d = ImageDraw.Draw(img)
    f = _font(15)
    for i in range(len(xs)):
        r, c = divmod(i, cols)
        x0, y0 = pad + c * (cell + pad), pad + r * (cell + pad + 18)
        img.paste(_tile(xs[i], cell), (x0, y0))
        d.rectangle([x0, y0, x0 + cell, y0 + cell], outline="#cc2222", width=2)
        d.text((x0, y0 + cell + 2), f"p={int(ps[i])} t={int(ts[i])}", fill="#222222", font=f)
    img.save(path)


def save_confusion(cm: np.ndarray, path: str) -> None:
    if Image is None:
        return
    cell, pad, left = 40, 2, 34
    n = cm.shape[0]
    img = Image.new("RGB", (left + n * (cell + pad) + pad, 34 + n * (cell + pad) + pad), "white")
    d = ImageDraw.Draw(img)
    f = _font(15)
    mx = max(1, cm.max())
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            shade = int(255 - 190 * (v / mx))
            x0, y0 = left + j * (cell + pad), 34 + i * (cell + pad)
            d.rectangle([x0, y0, x0 + cell, y0 + cell],
                        fill=(shade, shade, 255) if v else (246, 246, 246))
            if v:
                d.text((x0 + 7, y0 + 10), str(v), fill="#111111", font=f)
    for i in range(n):
        d.text((left + i * (cell + pad) + 13, 12), str(i), fill="#333333", font=f)
        d.text((8, 34 + i * (cell + pad) + 11), str(i), fill="#333333", font=f)
    d.text((left, 34 + n * (cell + pad) + 4), "rows=true  cols=pred", fill="#666666", font=f)
    img.save(path)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PyTorch CNN 训练 MNIST")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.05, help="标签平滑系数（0=关闭）")
    p.add_argument("--no-augment", action="store_true", help="关闭训练集随机旋转/平移增强")
    p.add_argument("--patience", type=int, default=0, help="连续 N 个 epoch 无提升则早停（0=关闭）")
    p.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / mps")
    p.add_argument("--data", type=str, default="data", help="MNIST 数据目录")
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--workers", type=int, default=2, help="DataLoader 进程数（Windows 上 0/2 较稳）")
    p.add_argument("--no-download", action="store_true", help="禁止联网下载 MNIST")
    p.add_argument("--eval-only", action="store_true", help="只评估已有权重")
    p.add_argument("--ckpt", type=str, default="outputs/mnist_cnn.pt")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.eval_only:
        device = pick_device(args.device)
        _, test_ds, source = load_mnist(args.data, download=False, augment=False)
        loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        model = MNISTCNN().to(device)
        model.load_state_dict(load_checkpoint(args.ckpt, device)["model"])
        loss, pred, target = evaluate_full(model, loader, device)
        print(f"测试集: loss {loss:.4f} | 准确率 {float((pred == target).mean()):.4f}")
        return 0

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_my_digit.py —— 用训练好的模型识别你自己手写的数字

这是验证 CNN 有没有真正学到东西的**试金石**：训练数据是别人写的字，
你自己的字对它来说是没见过的输入。

用法
    # 识别单张图片
    python predict_my_digit.py my_digit.png

    # 一次识别多张
    python predict_my_digit.py a.png b.png c.png

    # 显示每一张的前 3 个候选和概率
    python predict_my_digit.py my_digit.png --topk 3

    # 指定权重 / 强制反相 / 关掉居中
    python predict_my_digit.py my_digit.png --ckpt outputs/mnist_cnn.pt
    python predict_my_digit.py my_digit.png --invert
    python predict_my_digit.py my_digit.png --no-center

预处理为什么重要
    相机拍的照片、画图软件画的图，和 MNIST 的差别很大：
      * 背景可能是白的、数字是黑的（MNIST 是黑底白字）  -> 需要反相
      * 数字可能偏在一角、大小不一                        -> 需要按"质心"居中
    不做这两步，模型再准也会认错。--invert / --no-center 可以手动控制。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

DEFAULT_MEAN, DEFAULT_STD = 0.1307, 0.3081


# ------------------------------------------------------------------------------
# 预处理
# ------------------------------------------------------------------------------
def load_gray_28(path: str, center: bool = True, invert: str = "auto",
                 pad: int = 4) -> tuple[np.ndarray, dict]:
    """
    把任意图片变成 MNIST 风格的 28x28 灰度数组（值域 0~1，黑底白字）。

    invert: 'auto' 自动判断 | 'yes' 强制反相 | 'no' 不反相
    返回 (数组, 调试信息)
    """
    img = Image.open(path).convert("L")
    info: dict = {"原始尺寸": img.size}

    a = np.asarray(img, dtype=np.float32) / 255.0

    # --- 反相判断：MNIST 是黑底白字，即"背景暗、笔画亮" ---
    # 用四周边框的像素代表背景，比看整图均值更可靠（数字占比小）
    border = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    bg = float(border.mean())
    fg = float(a.mean())
    if invert == "yes":
        do_invert = True
    elif invert == "no":
        do_invert = False
    else:
        do_invert = bg > 0.5          # 背景偏亮 => 白底黑字 => 要反相
    if do_invert:
        a = 1.0 - a
    info["背景亮度"] = round(bg, 3)
    info["整图均值"] = round(fg, 3)
    info["已反相"] = do_invert

    # --- 裁掉空白边框，再按质心居中（MNIST 官方预处理就是这么做的）---
    mask = a > 0.15 * max(a.max(), 1e-6)
    if mask.any():
        ys, xs = np.where(mask)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        info["裁剪后"] = a.shape
    else:
        info["警告"] = "图片里几乎全是背景，可能找不到数字"

    # --- 等比缩放到 20x20（MNIST 数字大致占 20x20），再放进 28x28 画布 ---
    h, w = a.shape
    scale = 20.0 / max(h, w)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(new_size, Image.LANCZOS),
                   dtype=np.float32) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)
    if center and a.max() > 0:
        # 按亮度质心对齐，让数字落在画布中心
        ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        total = a.sum()
        cy, cx = float((ys * a).sum() / total), float((xs * a).sum() / total)
        top = int(round(14 - cy))
        left = int(round(14 - cx))
    else:
        top = (28 - a.shape[0]) // 2
        left = (28 - a.shape[1]) // 2
    top = max(0, min(28 - a.shape[0], top))
    left = max(0, min(28 - a.shape[1], left))
    canvas[top:top + a.shape[0], left:left + a.shape[1]] = a

    if canvas.max() > 0:
        canvas /= canvas.max()
    return canvas, info


# ------------------------------------------------------------------------------
# 两种后端
# ------------------------------------------------------------------------------
def predict_torch(arr: np.ndarray, ckpt: str, device: str = "auto") -> np.ndarray:
    import torch

    import cnn_mnist_torch as T

    dev = T.pick_device(device)
    model = T.MNISTCNN().to(dev)
    state = T.load_checkpoint(ckpt, dev)
    model.load_state_dict(state["model"])
    model.eval()

    x = torch.from_numpy(arr)[None, None].float()          # (1,1,28,28)
    x = (x - DEFAULT_MEAN) / DEFAULT_STD                   # 与训练时一致
    with torch.no_grad():
        probs = torch.softmax(model(x.to(dev)), dim=1)[0].cpu().numpy()
    return probs


def predict_numpy(arr: np.ndarray, ckpt: str) -> np.ndarray:
    import cnn_numpy as C

    with np.load(ckpt) as data:
        mean = float(data["mean"]) if "mean" in data else DEFAULT_MEAN
        std = float(data["std"]) if "std" in data else DEFAULT_STD
        param_names = [n for n in data.files if n.startswith("L")]
    model = C.SmallCNN(seed=0)
    expected = {name for name, _, _ in model.params()}
    if set(param_names) != expected:
        raise RuntimeError(f"权重与本版 NumPy 模型不匹配（{len(param_names)} vs {len(expected)} 个参数）")
    model.load_npz(ckpt)
    x = (arr[None, None] - mean) / std
    return C.softmax(model.forward(x))[0]


def find_available_ckpt(prefer: str | None) -> tuple[str, str] | None:
    """返回 (权重路径, 后端名)。优先 PyTorch 的真实 MNIST 权重。"""
    candidates = [
        (prefer, "torch" if (prefer or "").endswith(".pt") else "numpy"),
        ("outputs/mnist_cnn.pt", "torch"),
        ("outputs/numpy_cnn.npz", "numpy"),
    ]
    for path, backend in candidates:
        if path and os.path.exists(path):
            return path, backend
    return None


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def plot_result(arr: np.ndarray, probs: np.ndarray, title: str, out: str) -> None:
    """把 28x28 输入和概率条形图画成一张 PNG，方便肉眼核对。"""
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return
    W, H = 520, 150
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    big = Image.fromarray((arr * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST)
    img.paste(big.convert("RGB"), (10, 10))
    d.rectangle([10, 10, 138, 138], outline="#888888")

    top = int(probs.argmax())
    d.text((150, 8), f"{title}  ->  {top}   ({probs[top] * 100:.1f}%)", fill="#111111", font=font)
    for i in range(10):
        y = 30 + i * 11
        d.text((150, y), str(i), fill="#333333", font=font)
        w = int(probs[i] * 300)
        color = "#1f6fd0" if i != top else "#22aa44"
        d.rectangle([165, y, 165 + w, y + 9], fill=color)
        d.text((470, y), f"{probs[i] * 100:5.1f}%", fill="#333333", font=font)
    img.save(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="识别你自己手写的数字")
    p.add_argument("images", nargs="+", help="一张或多张图片路径")
    p.add_argument("--ckpt", type=str, default=None, help="权重路径（默认自动找 .pt 再找 .npz）")
    p.add_argument("--topk", type=int, default=1, help="显示前 K 个候选")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                   help="是否反相（白底黑字需要反相）")
    p.add_argument("--no-center", action="store_true", help="关掉质心居中")
    p.add_argument("--save-preview", action="store_true", help="把预处理结果和概率图存成 PNG")
    p.add_argument("--preview-dir", type=str, default="outputs")
    args = p.parse_args(argv)

    found = find_available_ckpt(args.ckpt)
    if found is None:
        print("找不到任何权重文件。请先训练：\n"
              "  python cnn_mnist_torch.py --epochs 5     (真实 MNIST)\n"
              "  python cnn_numpy.py --epochs 3           (离线版)")
        return 1
    ckpt, backend = found
    print(f"使用权重: {ckpt}  (后端: {backend})")

    if args.save_preview:
        os.makedirs(args.preview_dir, exist_ok=True)

    n_wrong = 0
    for path in args.images:
        if not os.path.exists(path):
            print(f"\n[跳过] 文件不存在: {path}")
            continue
        arr, info = load_gray_28(path, center=not args.no_center, invert=args.invert)

        if backend == "torch":
            probs = predict_torch(arr, ckpt, args.device)
        else:
            probs = predict_numpy(arr, ckpt)

        order = np.argsort(probs)[::-1][:args.topk]
        name = os.path.basename(path)
        print(f"\n--- {name} ---")
        for k, v in info.items():
            print(f"    {k}: {v}")
        print(f"    预测: {order[0]}  (置信度 {probs[order[0]] * 100:.1f}%)")
        if args.topk > 1:
            print("    候选: " + "  ".join(f"{i}:{probs[i] * 100:.1f}%" for i in order))

        # 置信度低时给个提示：多半是预处理或笔迹风格差异
        if probs[order[0]] < 0.5:
            print("    [提示] 置信度低于 50%，模型很犹豫 —— 可能图片太模糊、"
                  "数字没居中，或笔迹风格和 MNIST 差异过大")
        if args.save_preview:
            out = os.path.join(args.preview_dir, f"preview_{os.path.splitext(name)[0]}.png")
            plot_result(arr, probs, f"pred={order[0]}", out)
            print(f"    预览图: {out}")

    return 0 if n_wrong == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

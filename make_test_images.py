#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_test_images.py —— 从真实 MNIST 测试集导出几张图片，用来测试 predict_my_digit.py

导出成"白底黑字 + 四周留白"的形式，故意和 MNIST 原始格式（黑底白字）不同，
这样能顺带验证 predict_my_digit.py 的自动反相和居中预处理是否有效。

用法:
    python make_test_images.py                 # 导出 5 张到 my_digits/
    python make_test_images.py --count 8
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


def main() -> int:
    p = argparse.ArgumentParser(description="从 MNIST 测试集导出图片")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--outdir", type=str, default="my_digits")
    p.add_argument("--pad", type=int, default=40, help="四周留白像素")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # 直接用 torchvision 读（本机已装），保证是真实 MNIST
    from torchvision import datasets

    ds = datasets.MNIST("data", train=False, download=False,
                        transform=None)  # 拿原始 PIL 图
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=args.count, replace=False)

    print(f"从真实 MNIST 测试集导出 {args.count} 张到 {args.outdir}/")
    for i, idx in enumerate(idxs):
        img, label = ds[int(idx)]
        a = np.asarray(img, dtype=np.uint8)              # 黑底白字 28x28
        white_bg = 255 - a                                # 转成白底黑字
        canvas = np.full((28 + 2 * args.pad, 28 + 2 * args.pad), 255, dtype=np.uint8)
        canvas[args.pad:args.pad + 28, args.pad:args.pad + 28] = white_bg
        name = f"digit_{label}_{i}.png"
        Image.fromarray(canvas).resize(
            (canvas.shape[1] * 4, canvas.shape[0] * 4), Image.LANCZOS
        ).save(os.path.join(args.outdir, name))
        print(f"  {name}  (真实标签 {label}, 画布 {canvas.shape[1] * 4}x{canvas.shape[0] * 4})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

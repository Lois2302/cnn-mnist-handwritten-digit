#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_numpy.py —— 纯 NumPy 从零实现的卷积神经网络（CNN）

包含内容
    1. Conv2D        : 3x3 卷积，im2col + GEMM 实现，含完整反向传播
    2. MaxPool2D     : 2x2 最大池化，含 argmax 路由的反向传播
    3. ReLU / Dense  : 激活层与全连接层
    4. Softmax + 交叉熵损失（含梯度）
    5. 程序化数字数据集：用 PIL 渲染带随机旋转/平移/缩放/粗细的手写风格数字
       （在没有网络、装不上 torch 的机器上，仍然可以真实训练一个 CNN）
    6. 训练循环、验证、混淆矩阵、预测样例图，以及 --grad-check 数值梯度校验

为什么有程序化数据集？
    标准 MNIST 需要联网下载（torchvision 会自动下载）。本机处于无网受限环境，
    因此这份实现自带一个"用字体渲染出来的数字"数据集，可以离线端到端跑通。
    想用真实 MNIST，请运行 cnn_mnist_torch.py。

快速开始
    python cnn_numpy.py --epochs 3              # 真实训练
    python cnn_numpy.py --grad-check            # 数值梯度校验（验证反向传播写对了）
    python cnn_numpy.py --epochs 3 --no-rasterize   # 纯 NumPy 渲染，速度更快

依赖: numpy, Pillow
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageFilter = ImageFont = None

EPS = 1e-12

# 优先尝试的等宽/衬线字体，尽量覆盖中英文 Windows 环境
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\georgia.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\trebuc.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


# ------------------------------------------------------------------------------
# 1. 基础层
# ------------------------------------------------------------------------------
class Layer:
    """所有层的基类：统一 forward / backward / params 接口。"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """返回 [(名字, 参数, 梯度), ...]，无参数层返回空列表。"""
        return []


def he_normal(shape: tuple[int, ...], fan_in: int, rng: np.random.Generator) -> np.ndarray:
    """He 初始化，适合 ReLU 网络。"""
    return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)


class Conv2D(Layer):
    """
    二维卷积层（不带 padding，stride=1）。
    forward 用 im2col 把卷积变成矩阵乘法，所以速度可以接受。

    张量布局: (N, C, H, W)
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.in_ch, self.out_ch, self.k = in_ch, out_ch, k
        self.W = he_normal((out_ch, in_ch, k, k), in_ch * k * k, rng)
        self.b = np.zeros(out_ch, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache: dict[str, Any] = {}

    @staticmethod
    def _im2col(x: np.ndarray, k: int) -> np.ndarray:
        """(N,C,H,W) -> (N, C*k*k, H-k+1, W-k+1) 的列块视图（视图拷贝，便于广播）。"""
        n, c, h, w = x.shape
        oh, ow = h - k + 1, w - k + 1
        s = x.strides
        view = np.lib.stride_tricks.as_strided(
            x,
            shape=(n, c, oh, ow, k, k),
            strides=(s[0], s[1], s[2], s[3], s[2], s[3]),
            writeable=False,
        )
        # (N, oh, ow, C, k, k) -> (N, oh*ow, C*k*k)
        return np.ascontiguousarray(view.transpose(0, 2, 3, 1, 4, 5)).reshape(n, oh * ow, c * k * k)

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        assert c == self.in_ch, f"通道数不匹配: 期望 {self.in_ch}, 实际 {c}"
        k = self.k
        oh, ow = h - k + 1, w - k + 1
        cols = self._im2col(x, k)                                  # (N, oh*ow, C*k*k)
        wmat = self.W.reshape(self.out_ch, -1)                     # (O, C*k*k)
        out = cols @ wmat.T + self.b                               # (N, oh*ow, O)
        out = out.transpose(0, 2, 1).reshape(n, self.out_ch, oh, ow)
        self._cache = {"cols": cols, "x_shape": x.shape, "oh": oh, "ow": ow}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        cache = self._cache
        cols, (n, _, h, w) = cache["cols"], cache["x_shape"]
        oh, ow = cache["oh"], cache["ow"]

        dout_flat = dout.transpose(0, 2, 3, 1).reshape(n, oh * ow, self.out_ch)
        # 梯度: dW = sum_n dout^T @ cols ；db = sum_n dout
        self.dW = np.tensordot(dout_flat, cols, axes=([0, 1], [0, 1])).reshape(self.W.shape)
        self.db = dout_flat.sum(axis=(0, 1))

        # 把梯度散回输入空间（col2im）
        dcols = dout_flat @ self.W.reshape(self.out_ch, -1)         # (N, oh*ow, C*k*k)
        dx = np.zeros((n, self.in_ch, h, w), dtype=np.float64)
        dcols = dcols.reshape(n, oh, ow, self.in_ch, self.k, self.k).transpose(0, 3, 1, 2, 4, 5)
        for i in range(self.k):
            for j in range(self.k):
                dx[:, :, i:i + oh, j:j + ow] += dcols[:, :, :, :, i, j]
        return dx

    def params(self):
        return [("conv.W", self.W, self.dW), ("conv.b", self.b, self.db)]


class MaxPool2D(Layer):
    """2x2 最大池化（stride=2），反向传播只把梯度回传给当初取到最大值的位置。"""

    def __init__(self, size: int = 2):
        self.size = size
        self._argmax: np.ndarray | None = None
        self._in_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        s = self.size
        self._in_shape = (n, c, h, w)
        oh, ow = h // s, w // s
        h2, w2 = oh * s, ow * s
        x = x[:, :, :h2, :w2]           # 奇数边长的最后一行/列会被丢弃
        view = x.reshape(n, c, oh, s, ow, s).transpose(0, 1, 2, 4, 3, 5)
        flat = view.reshape(n, c, oh, ow, s * s)
        self._argmax = flat.argmax(axis=-1)
        return flat.max(axis=-1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n, c, h, w = self._in_shape
        s = self.size
        oh, ow = h // s, w // s
        dx = np.zeros((n, c, oh, ow, s * s), dtype=np.float64)
        np.put_along_axis(dx, self._argmax[..., None], dout[..., None], axis=-1)
        # (N, C, oh, ow, s, s) --transpose--> (N, C, oh, s, ow, s)
        dx = dx.reshape(n, c, oh, ow, s, s).transpose(0, 1, 2, 4, 3, 5)
        dx = dx.reshape(n, c, oh * s, ow * s)
        # 被丢弃的最后一行/列对应的梯度必须是 0，所以要还原成原始输入尺寸
        if oh * s != h or ow * s != w:
            full = np.zeros((n, c, h, w), dtype=np.float64)
            full[:, :, :oh * s, :ow * s] = dx
            dx = full
        return dx

    def params(self):
        return []


class ReLU(Layer):
    def __init__(self):
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask

    def params(self):
        return []


class Flatten(Layer):
    def __init__(self):
        self._shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._shape)

    def params(self):
        return []


class Dense(Layer):
    """全连接层。"""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator | None = None,
                 init_scale: float = 1.0):
        rng = rng or np.random.default_rng(0)
        self.W = he_normal((in_dim, out_dim), in_dim, rng) * init_scale
        self.b = np.zeros(out_dim, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self):
        return [("dense.W", self.W, self.dW), ("dense.b", self.b, self.db)]


# ------------------------------------------------------------------------------
# 2. 损失函数
# ------------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)      # 数值稳定
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + EPS)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    n = probs.shape[0]
    return float(-np.log(probs[np.arange(n), y] + EPS).mean())


def cross_entropy_grad(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """softmax + 交叉熵 对 logits 的梯度（最漂亮的那一步）。"""
    n = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    return grad / n


def one_hot(y: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros((y.size, classes), dtype=np.float64)
    out[np.arange(y.size), y] = 1.0
    return out


# ------------------------------------------------------------------------------
# 3. 小型 CNN 模型
# ------------------------------------------------------------------------------
class SmallCNN:
    """
    结构（默认输入 1x28x28，通道/尺寸都由下面参数推导，不写死）:
        Conv(1->8, 3x3)  -> ReLU -> MaxPool(2)    : 28 -> 26 -> 13
        Conv(8->16, 3x3) -> ReLU -> MaxPool(2)    : 13 -> 11 -> 5
        Flatten                                    : 16*5*5 = 400
        Dense(400->64) -> ReLU
        Dense(64->num_classes)
    """

    def __init__(self, num_classes: int = 10, seed: int = 0, in_ch: int = 1,
                 in_size: int = 28, c1: int = 8, c2: int = 16, hidden: int = 64):
        rng = np.random.default_rng(seed)
        self.num_classes = num_classes
        self.in_ch, self.in_size = in_ch, in_size
        self.conv1 = Conv2D(in_ch, c1, 3, rng)
        self.conv2 = Conv2D(c1, c2, 3, rng)

        # 逐层推导空间尺寸，避免手算错误
        size = in_size
        size = size - 3 + 1          # conv1
        size //= 2                   # pool1
        size = size - 3 + 1          # conv2
        size //= 2                   # pool2
        if size <= 0:
            raise ValueError("输入尺寸太小，无法通过两层 3x3 卷积 + 2x2 池化")
        self.flat_dim = c2 * size * size

        self.layers: list[Layer] = [
            # 注意：每个层必须是独立实例。层对象内部会缓存 forward 的中间结果
            # （ReLU 的 mask、MaxPool 的 argmax），复用同一个实例会互相覆盖。
            self.conv1, ReLU(), MaxPool2D(2),
            self.conv2, ReLU(), MaxPool2D(2),
            Flatten(),
            Dense(self.flat_dim, hidden, rng), ReLU(),
            Dense(hidden, num_classes, rng),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dlogits: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dlogits = layer.backward(dlogits)

    def params(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """参数名唯一（同一层可能出现多次，所以带上层序号）。"""
        out: list[tuple[str, np.ndarray, np.ndarray]] = []
        for i, layer in enumerate(self.layers):
            for name, p, g in layer.params():
                out.append((f"L{i}_{name}", p, g))
        return out

    def predict(self, x: np.ndarray, batch: int = 256) -> np.ndarray:
        preds = []
        for i in range(0, len(x), batch):
            logits = self.forward(x[i:i + batch])
            preds.append(logits.argmax(axis=1))
        return np.concatenate(preds)

    def save_npz(self, path: str) -> None:
        np.savez(path, **{name: p for name, p, _ in self.params()})

    def load_npz(self, path: str) -> None:
        with np.load(path) as data:
            for name, p, _ in self.params():
                p[...] = data[name]


# ------------------------------------------------------------------------------
# 4. 优化器（带动量的 SGD）
# ------------------------------------------------------------------------------
class SGD:
    def __init__(self, model: SmallCNN, lr: float = 3e-3, momentum: float = 0.9,
                 weight_decay: float = 1e-4):
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.velocity = {name: np.zeros_like(p) for name, p, _ in model.params()}

    def step(self, model: SmallCNN) -> None:
        for name, p, g in model.params():
            grad = g + self.wd * p
            v = self.velocity[name]
            v *= self.momentum
            v -= self.lr * grad
            p += v


# ------------------------------------------------------------------------------
# 5. 程序化数字数据集
# ------------------------------------------------------------------------------
def pick_font(size: int) -> Any:
    """挑一个可用的系统字体；找不到就退化为 PIL 内置位图字体。"""
    if ImageFont is None:
        raise RuntimeError("需要 Pillow: pip install pillow")
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # 老版本 Pillow
        return ImageFont.load_default()


def _render_digit(digit: int, font: Any, rng: np.random.Generator, canvas: int = 28,
                  supersample: int = 2) -> np.ndarray:
    """把一个数字渲染成 canvas x canvas 的 float32 灰度图（值域 0~1）。"""
    big = canvas * supersample
    img = Image.new("L", (big, big), color=0)
    draw = ImageDraw.Draw(img)

    # 随机字号
    base_size = int(big * rng.uniform(0.55, 0.80))
    try:
        f = ImageFont.truetype(font.path, base_size) if hasattr(font, "path") else font
    except Exception:
        f = font

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ox = (big - tw) / 2 - bbox[0]
    oy = (big - th) / 2 - bbox[1]

    # 随机笔画粗细（模拟不同笔压）
    stroke = int(max(1, round(big * rng.uniform(0.02, 0.055))))
    draw.text((ox, oy), text, fill=255, font=f, stroke_width=stroke, stroke_fill=255)

    # 随机仿射：旋转 / 缩放 / 平移 / 轻微错切
    angle = rng.uniform(-18, 18)
    scale = rng.uniform(0.85, 1.12)
    shear = rng.uniform(-0.15, 0.15)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    img = img.transform(
        (big, big), Image.AFFINE,
        (1 / scale, shear, -shear * big / 2, 0, 1 / scale, -(1 / scale - 1) * big / 2),
        resample=Image.BILINEAR, fillcolor=0,
    )
    pad = int(big * 0.08)
    img = img.crop((pad, pad, big - pad, big - pad)).resize((canvas, canvas), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.max() > 0:
        arr /= arr.max()
    # 轻微高斯噪声，避免网络死记硬背
    arr = np.clip(arr + rng.normal(0, 0.03, arr.shape), 0, 1)
    return arr


def _rasterize_batch(digit: int, count: int, rng: np.random.Generator, canvas: int = 28,
                     font_sizes: tuple[int, ...] = (20, 24, 30)) -> np.ndarray:
    """用字体把某个数字渲染 count 次，得到 (count, 1, canvas, canvas)。"""
    out = np.empty((count, 1, canvas, canvas), dtype=np.float64)
    for i in range(count):
        font = pick_font(int(font_sizes[i % len(font_sizes)]))
        out[i, 0] = _render_digit(digit, font, rng, canvas)
    return out


def _strokes(digit: int, rng: np.random.Generator, canvas: int = 28, samples: int = 220) -> np.ndarray:
    """
    纯 NumPy 的"笔画"渲染（不需要字体）：把每个数字定义成若干折线段，
    再随机抖动端点、加随机粗细。作为 Pillow 缺失或 --no-rasterize 时的后备方案。
    """
    img = np.zeros((canvas, canvas), dtype=np.float64)
    # 每个数字的笔画：[(起点, 终点, ...), ...]，坐标是 0~1 归一化
    shapes: dict[int, list[list[tuple[float, float]]]] = {
        0: [[(0.30, 0.20), (0.70, 0.20), (0.78, 0.50), (0.70, 0.80), (0.30, 0.80), (0.22, 0.50), (0.30, 0.20)]],
        1: [[(0.42, 0.15), (0.55, 0.10), (0.55, 0.85)], [(0.38, 0.85), (0.72, 0.85)]],
        2: [[(0.25, 0.28), (0.50, 0.15), (0.75, 0.30), (0.68, 0.52), (0.28, 0.80), (0.25, 0.85), (0.78, 0.85)]],
        3: [[(0.26, 0.22), (0.66, 0.18), (0.74, 0.42), (0.48, 0.50), (0.76, 0.62), (0.68, 0.85), (0.26, 0.80)]],
        4: [[(0.62, 0.15), (0.26, 0.62), (0.78, 0.62)], [(0.62, 0.15), (0.62, 0.86)]],
        5: [[(0.74, 0.18), (0.32, 0.18), (0.28, 0.45), (0.58, 0.42), (0.76, 0.60), (0.66, 0.85), (0.28, 0.80)]],
        6: [[(0.68, 0.18), (0.38, 0.32), (0.26, 0.62), (0.36, 0.84), (0.66, 0.82), (0.72, 0.60), (0.50, 0.52), (0.28, 0.60)]],
        7: [[(0.24, 0.18), (0.76, 0.18), (0.48, 0.86)]],
        8: [[(0.50, 0.16), (0.30, 0.30), (0.50, 0.48), (0.70, 0.32), (0.50, 0.16)],
            [(0.50, 0.48), (0.26, 0.68), (0.50, 0.86), (0.74, 0.68), (0.50, 0.48)]],
        9: [[(0.72, 0.40), (0.50, 0.20), (0.30, 0.32), (0.34, 0.56), (0.58, 0.60), (0.72, 0.44)], [(0.66, 0.48), (0.60, 0.86)]],
    }
    jitter = rng.normal(0, 0.035, (2,))
    for poly in shapes[digit]:
        pts = np.array(poly, dtype=np.float64) * (canvas - 1)
        pts = pts + rng.normal(0, 0.9, pts.shape) + jitter * (canvas - 1)
        width = rng.uniform(1.4, 2.6)
        for a, b in zip(pts[:-1], pts[1:]):
            for t in np.linspace(0, 1, max(2, samples // max(1, len(pts) - 1))):
                p = a + t * (b - a)
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < canvas and 0 <= y < canvas:
                    img[max(0, y - int(width)):y + int(width) + 1,
                        max(0, x - int(width)):x + int(width) + 1] = 1.0
    # 轻微模糊，模拟真实笔迹边缘
    uint8 = Image.fromarray((img * 255).astype(np.uint8))
    uint8 = uint8.filter(ImageFilter.GaussianBlur(0.6))
    return np.asarray(uint8, dtype=np.float64) / 255.0


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"[{self.source}] 训练集 {self.x_train.shape} / 测试集 {self.x_test.shape} "
                f"类别 {sorted(set(self.y_train.tolist()))}")


def make_digit_dataset(n_train: int = 4000, n_test: int = 800, seed: int = 0,
                       rasterize: bool = True, canvas: int = 28) -> Dataset:
    """生成程序化数字数据集，类别均衡。"""
    rng = np.random.default_rng(seed)
    per_train = n_train // 10
    per_test = max(1, n_test // 10)

    if rasterize and Image is not None:
        try:
            xtr = np.concatenate([_rasterize_batch(d, per_train, rng, canvas) for d in range(10)])
            xte = np.concatenate([_rasterize_batch(d, per_test, rng, canvas) for d in range(10)])
            src = "procedural-font"
        except Exception as exc:  # 字体渲染失败就退到笔画版
            print(f"[warn] 字体渲染失败({exc})，改用笔画渲染", file=sys.stderr)
            rasterize = False
    if not rasterize or Image is None:
        xtr = np.concatenate([
            np.stack([_strokes(d, rng, canvas) for _ in range(per_train)])[:, None] for d in range(10)])
        xte = np.concatenate([
            np.stack([_strokes(d, rng, canvas) for _ in range(per_test)])[:, None] for d in range(10)])
        src = "procedural-stroke"

    ytr = np.repeat(np.arange(10), per_train)
    yte = np.repeat(np.arange(10), per_test)
    # 打乱
    p_tr, p_te = rng.permutation(len(ytr)), rng.permutation(len(yte))
    return Dataset(xtr[p_tr], ytr[p_tr], xte[p_te], yte[p_te], source=src,
                   meta={"canvas": canvas, "seed": seed})


def normalize(ds: Dataset) -> tuple[float, float]:
    """全局标准化，返回 (mean, std) 方便推理时复用。"""
    mean = float(ds.x_train.mean())
    std = float(ds.x_train.std()) + EPS
    for split in ("x_train", "x_test"):
        arr = getattr(ds, split)
        setattr(ds, split, (arr - mean) / std)
    return mean, std


# ------------------------------------------------------------------------------
# 6. 训练 / 评估
# ------------------------------------------------------------------------------
@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


def iterate_batches(n: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for i in range(0, n, batch_size):
        yield order[i:i + batch_size]


def train(model: SmallCNN, ds: Dataset, epochs: int = 3, batch_size: int = 64,
          lr: float = 3e-3, seed: int = 0, verbose: bool = True) -> History:
    rng = np.random.default_rng(seed)
    opt = SGD(model, lr=lr)
    hist = History()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        losses, correct, seen = [], 0, 0
        for idx in iterate_batches(len(ds.y_train), batch_size, rng):
            xb, yb = ds.x_train[idx], ds.y_train[idx]
            probs = softmax(model.forward(xb))
            loss = cross_entropy(probs, yb)
            model.backward(cross_entropy_grad(probs, yb))
            opt.step(model)

            losses.append(loss)
            correct += int((probs.argmax(axis=1) == yb).sum())
            seen += len(yb)

        tr_loss, tr_acc = float(np.mean(losses)), correct / seen
        vl_loss, vl_acc = evaluate(model, ds.x_test, ds.y_test, batch_size)
        hist.train_loss.append(tr_loss)
        hist.train_acc.append(tr_acc)
        hist.val_loss.append(vl_loss)
        hist.val_acc.append(vl_acc)
        hist.seconds.append(time.time() - t0)
        if verbose:
            print(f"epoch {epoch}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
                  f"| val loss {vl_loss:.4f} acc {vl_acc:.4f} | {hist.seconds[-1]:.1f}s")
    return hist


def evaluate(model: SmallCNN, x: np.ndarray, y: np.ndarray, batch_size: int = 256) -> tuple[float, float]:
    losses, correct = [], 0
    for i in range(0, len(y), batch_size):
        xb, yb = x[i:i + batch_size], y[i:i + batch_size]
        probs = softmax(model.forward(xb))
        losses.append(cross_entropy(probs, yb) * len(yb))
        correct += int((probs.argmax(axis=1) == yb).sum())
    return float(np.sum(losses) / len(y)), correct / len(y)


def confusion_matrix(pred: np.ndarray, y: np.ndarray, classes: int = 10) -> np.ndarray:
    cm = np.zeros((classes, classes), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


# ------------------------------------------------------------------------------
# 7. 数值梯度校验：证明反向传播没写错
# ------------------------------------------------------------------------------
def grad_check(model: SmallCNN, batch: int = 4, canvas: int = 28,
               num_checks: int = 6, eps: float = 1e-5, seed: int = 0) -> float:
    """有限差分对比解析梯度，返回最大相对误差。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch, 1, canvas, canvas))
    y = rng.integers(0, model.num_classes, batch)

    probs = softmax(model.forward(x))
    model.backward(cross_entropy_grad(probs, y))

    def loss_of() -> float:
        return cross_entropy(softmax(model.forward(x)), y)

    worst = 0.0
    checked = 0
    for name, p, g in model.params():
        flat_p, flat_g = p.reshape(-1), g.reshape(-1)
        for k in rng.choice(flat_p.size, size=min(num_checks, flat_p.size), replace=False):
            orig = flat_p[k]
            flat_p[k] = orig + eps
            lp = loss_of()
            flat_p[k] = orig - eps
            lm = loss_of()
            flat_p[k] = orig
            num = (lp - lm) / (2 * eps)
            ana = flat_g[k]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            checked += 1
    print(f"数值梯度校验: 检查了 {checked} 个参数，最大相对误差 {worst:.3e}")
    return worst


# ------------------------------------------------------------------------------
# 8. 结果可视化（不需要 matplotlib，直接用 Pillow 画图）
# ------------------------------------------------------------------------------
def save_history_plot(hist: History, path: str) -> None:
    if Image is None:
        return
    w, h, m = 720, 300, 40
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = pick_font(13)

    def poly(values: list[float], color: str, vmax: float, vmin: float) -> None:
        pts = []
        for i, v in enumerate(values):
            px = m + i * (w - 2 * m) / max(1, len(values) - 1)
            py = h - m - (v - vmin) / max(1e-9, vmax - vmin) * (h - 2 * m)
            pts.append((px, py))
        if len(pts) > 1:
            d.line(pts, fill=color, width=2)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)

    d.rectangle([m, m, w - m, h - m], outline="#cccccc")
    d.text((m, 12), "loss (blue=train, red=val)", fill="#333333", font=font)
    all_loss = hist.train_loss + hist.val_loss
    poly(hist.train_loss, "#1f6fd0", max(all_loss) * 1.1, 0)
    poly(hist.val_loss, "#d03a1f", max(all_loss) * 1.1, 0)
    d.text((m, h - 22), f"val acc: {['%.3f' % a for a in hist.val_acc]}", fill="#333333", font=font)
    img.save(path)


def save_prediction_grid(model: SmallCNN, x: np.ndarray, y: np.ndarray, path: str,
                         cols: int = 8, rows: int = 4, canvas: int = 28) -> None:
    if Image is None:
        return
    n = min(cols * rows, len(y))
    idx = np.linspace(0, len(y) - 1, n).astype(int)
    probs = softmax(model.forward(x[idx]))
    pred = probs.argmax(axis=1)
    cell, pad = canvas * 3, 6
    img = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + pad + 16) + pad), "white")
    d = ImageDraw.Draw(img)
    font = pick_font(16)
    for i in range(n):
        r, c = divmod(i, cols)
        x0, y0 = pad + c * (cell + pad), pad + r * (cell + pad + 16)
        arr = (np.clip(x[idx[i], 0] * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
        tile = Image.fromarray(arr).resize((cell, cell), Image.NEAREST).convert("RGB")
        img.paste(tile, (x0, y0))
        ok = pred[i] == y[idx[i]]
        d.rectangle([x0, y0, x0 + cell, y0 + cell], outline="#22aa44" if ok else "#cc2222", width=2)
        d.text((x0, y0 + cell + 1), f"p={pred[i]} t={y[idx[i]]}", fill="#222222", font=font)
    img.save(path)


def save_confusion(cm: np.ndarray, path: str) -> None:
    if Image is None:
        return
    cell, pad, left = 34, 2, 30
    n = cm.shape[0]
    img = Image.new("RGB", (left + n * (cell + pad) + pad, 30 + n * (cell + pad) + pad), "white")
    d = ImageDraw.Draw(img)
    font = pick_font(14)
    mx = max(1, cm.max())
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            shade = int(255 - 200 * (v / mx))
            x0, y0 = left + j * (cell + pad), 30 + i * (cell + pad)
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=(shade, shade, 255) if v else (245, 245, 245))
            if v:
                d.text((x0 + 6, y0 + 8), str(v), fill="#111111", font=font)
    for i in range(n):
        d.text((left + i * (cell + pad) + 10, 10), str(i), fill="#333333", font=font)
        d.text((6, 30 + i * (cell + pad) + 8), str(i), fill="#333333", font=font)
    d.text((left, 30 + n * (cell + pad) + 6), "rows=true cols=pred", fill="#666666", font=font)
    img.save(path)


def save_checkpoint(model: SmallCNN, path: str, mean: float, std: float, hist: History) -> None:
    """保存权重 + 预处理参数；权重名与 model.load_npz 对应。"""
    arrays: dict[str, np.ndarray] = {"mean": np.array(mean), "std": np.array(std),
                                     "val_acc": np.array(hist.val_acc)}
    arrays.update({name: p for name, p, _ in model.params()})
    np.savez(path, **arrays)


# ------------------------------------------------------------------------------
# 9. CLI
# ------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="纯 NumPy CNN（程序化数字数据集，可离线训练）")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--train-size", type=int, default=4000)
    p.add_argument("--test-size", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--grad-check", action="store_true", help="只做数值梯度校验")
    p.add_argument("--no-rasterize", action="store_true", help="用纯 NumPy 笔画渲染代替字体渲染")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    model = SmallCNN(seed=args.seed)
    if args.grad_check:
        worst = grad_check(model, seed=args.seed)
        return 0 if worst < 1e-4 else 1

    print("== 构建数据集 ==")
    ds = make_digit_dataset(args.train_size, args.test_size, seed=args.seed,
                            rasterize=not args.no_rasterize)
    mean, std = normalize(ds)
    print(ds.summary())
    print(f"标准化: mean={mean:.4f} std={std:.4f}")

    print("\n== 训练 ==")
    hist = train(model, ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                 seed=args.seed, verbose=not args.quiet)

    print("\n== 评估 ==")
    pred = model.predict(ds.x_test)
    acc = float((pred == ds.y_test).mean())
    cm = confusion_matrix(pred, ds.y_test)
    print(f"测试集准确率: {acc:.4f}")
    print("混淆矩阵 (行=真实, 列=预测):")
    print(cm)

    save_history_plot(hist, os.path.join(args.outdir, "numpy_history.png"))
    save_prediction_grid(model, ds.x_test, ds.y_test, os.path.join(args.outdir, "numpy_predictions.png"))
    save_confusion(cm, os.path.join(args.outdir, "numpy_confusion.png"))
    save_checkpoint(model, os.path.join(args.outdir, "numpy_cnn.npz"), mean, std, hist)
    print(f"\n结果已写入 {args.outdir}/ : numpy_history.png, numpy_predictions.png, "
          f"numpy_confusion.png, numpy_cnn.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

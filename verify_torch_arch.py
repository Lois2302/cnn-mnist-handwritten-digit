#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_torch_arch.py —— 在没有安装 torch 的机器上验证 cnn_mnist_torch.py 的网络结构

原理
    cnn_mnist_torch.py 无法在本机运行（torch 装不上），但它的**张量维度自洽性**
    可以脱离 torch 验证：这里用 NumPy 写一个与 MNISTCNN.features / classifier
    逐层一一对应的镜像，按同样的假设推导形状，检查维度是否串得起来。

    顺带比对 MNISTCNN 的源码：确保镜像没有和源码脱节（层清单一致）。

用法:
    python verify_torch_arch.py
"""

from __future__ import annotations

import ast
import os
import sys

import numpy as np

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
TORCH_FILE = os.path.join(WORKSPACE, "cnn_mnist_torch.py")

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


# --- NumPy 镜像：与 MNISTCNN 的层顺序严格对应 -------------------------------------
def mirror_forward(n: int = 2, h: int = 28, w: int = 28, verbose: bool = True) -> tuple[int, list]:
    """按 MNISTCNN 的层顺序推导尺寸，返回 (展平维度, 每层尺寸记录)。"""
    x = np.zeros((n, 1, h, w), dtype=np.float32)
    log: list[tuple[str, tuple]] = []
    k = 3

    def conv(t, cin, cout, pad=1):
        _, _, H, W = t.shape
        out = np.zeros((t.shape[0], cout, H + 2 * pad - k + 1, W + 2 * pad - k + 1), np.float32)
        log.append((f"Conv2d({cin}->{cout},3,pad=1)", out.shape))
        return out

    def bn(t):
        log.append(("BatchNorm2d", t.shape))
        return t

    def relu(t):
        log.append(("ReLU", t.shape))
        return t

    def pool(t):
        out = np.zeros((t.shape[0], t.shape[1], t.shape[2] // 2, t.shape[3] // 2), np.float32)
        log.append(("MaxPool2d(2)", out.shape))
        return out

    def dropout2d(t):
        log.append(("Dropout2d", t.shape))
        return t

    def gap(t, size):
        out = np.zeros((t.shape[0], t.shape[1], size, size), np.float32)
        log.append((f"AdaptiveAvgPool2d({size})", out.shape))
        return out

    def flatten(t):
        out = t.reshape(t.shape[0], -1)
        log.append(("Flatten", out.shape))
        return out

    def linear(t, dim):
        out = np.zeros((t.shape[0], dim), np.float32)
        log.append((f"Linear({t.shape[1]}->{dim})", out.shape))
        return out

    # features[0..4]: conv/bn/relu/conv/bn/relu/pool/dropout
    x = conv(x, 1, 32); x = bn(x); x = relu(x)
    x = conv(x, 32, 32); x = bn(x); x = relu(x)
    x = pool(x); x = dropout2d(x)
    # features[8..13]
    x = conv(x, 32, 64); x = bn(x); x = relu(x)
    x = conv(x, 64, 64); x = bn(x); x = relu(x)
    x = pool(x); x = dropout2d(x)
    # 分类头
    x = gap(x, 3)
    x = flatten(x)
    flat_dim = int(x.shape[1])
    x = linear(x, 128); x = relu(x)
    x = linear(x, 10)

    if verbose:
        print(f"   输入 (2, 1, {h}, {w})")
        for name, shape in log:
            print(f"   {name:<26} -> {shape}")
    return flat_dim, log


def _const(node: ast.AST):
    """把 AST 节点求值成 Python 常量（支持 64 * 3 * 3 这种表达式和元组）。"""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = _const(node.left), _const(node.right)
        if isinstance(left, int) and isinstance(right, int):
            return left * right
    return None


def parse_model_source(path: str) -> dict:
    """从源码里抽出真实声明的层，用于和镜像比对。"""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    info = {"conv_out": [], "linear": [], "gap": None, "has_bn": False, "has_dropout": False}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        args = [_const(a) for a in node.args]
        if fname == "Conv2d" and len(args) >= 2 and all(isinstance(a, int) for a in args[:2]):
            info["conv_out"].append((args[0], args[1]))
        elif fname == "Linear" and len(node.args) >= 1:
            out_dim = args[1] if len(args) >= 2 and isinstance(args[1], int) else None
            if isinstance(args[0], int):
                # 输出维度写成变量（如 num_classes）时记为 None，仅校验输入维度
                info["linear"].append((args[0], out_dim))
        elif fname == "AdaptiveAvgPool2d" and args:
            first = args[0]
            info["gap"] = tuple(first) if isinstance(first, tuple) else first
        elif fname == "BatchNorm2d":
            info["has_bn"] = True
        elif fname in ("Dropout", "Dropout2d"):
            info["has_dropout"] = True
    return info


def main() -> int:
    print("=" * 62)
    print("验证 cnn_mnist_torch.py 的网络结构（无需 torch）")
    print("=" * 62)

    if not os.path.exists(TORCH_FILE):
        print(f"找不到 {TORCH_FILE}")
        return 1

    print("\n== 1. 逐层尺寸推演 ==")
    flat_dim, log = mirror_forward()

    print("\n== 2. 源码声明比对 ==")
    info = parse_model_source(TORCH_FILE)
    print(f"   源码 Conv2d 输出通道: {[c for _, c in info['conv_out']]}")
    print(f"   源码 Linear 层:       {info['linear']}")
    print(f"   源码 AdaptiveAvgPool: {info['gap']}")
    check("源码有 4 个 Conv2d", len(info["conv_out"]) == 4, str(len(info["conv_out"])))
    check("Conv 通道 32->32->64->64",
          [c for _, c in info["conv_out"]] == [32, 32, 64, 64], str(info["conv_out"]))
    check("有 BatchNorm", info["has_bn"])
    check("有 Dropout", info["has_dropout"])
    check("AdaptiveAvgPool2d 为 (3,3)", info["gap"] == (3, 3), str(info["gap"]))

    print("\n== 3. 关键维度自洽性 ==")
    check("自适应池化后通道数 64", log[-5][1][1] == 64, str(log[-5]))
    check("自适应池化后尺寸 3x3", log[-5][1][2:] == (3, 3), str(log[-5]))
    check(f"展平维度 = 64*3*3 = 576", flat_dim == 576, str(flat_dim))

    expected_linear = [(flat_dim, 128), (128, None)]
    check("Linear 层与展平维度匹配（输出维度写变量的记为 None）",
          info["linear"] == expected_linear,
          f"源码 {info['linear']} vs 期望 {expected_linear}")

    print("\n== 4. 不同输入尺寸的鲁棒性（自适应池化的意义）==")
    for size in (28, 32, 64):
        fd, _ = mirror_forward(h=size, w=size, verbose=False)
        print(f"   输入 {size}x{size} -> 展平 {fd}")
        check(f"输入 {size}x{size} 展平维度仍为 576", fd == 576, str(fd))

    print("\n== 5. 参数量估算 ==")
    # 卷积: 32*1*9 + 32*32*9 + 64*32*9 + 64*64*9 ; BN 参数 2*(32+32+64+64)
    # 全连接: 576*128+128 + 128*10+10
    # 注意：这里只是静态估算，真实值以训练日志为准（实测 140,522）
    conv = 32 * 1 * 9 + 32 * 32 * 9 + 64 * 32 * 9 + 64 * 64 * 9
    bn = 2 * (32 + 32 + 64 + 64)
    fc = 576 * 128 + 128 + 128 * 10 + 10
    total = conv + bn + fc
    print(f"   卷积 {conv:,} + BN {bn:,} + 全连接 {fc:,} = {total:,}（静态估算）")
    check("估算值在 1e5 ~ 1e6 之间", 1e5 < total < 1e6, f"{total:,}")

    print("\n" + "=" * 62)
    if FAILED:
        print(f"结果: {len(FAILED)} 项失败 -> {', '.join(FAILED)}")
        return 1
    print("结果: 全部通过（结构维度自洽；未在本机执行 torch 前向）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

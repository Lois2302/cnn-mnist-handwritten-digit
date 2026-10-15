#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_test.py —— CNN 实现的快速自检（秒级跑完）

检查项
    1. 各层 forward 的形状是否正确（逐层打印）
    2. 池化反向传播是否把被丢弃的奇数行/列还原成 0
    3. 数值梯度校验（有限差分 vs 解析梯度）
    4. 过拟合单批小数据：能否把 10 个样本训练到 100% 正确
    5. 短训练后准确率是否显著高于随机猜测（1/10）

用法:
    python smoke_test.py
"""

from __future__ import annotations

import sys

import numpy as np

import cnn_numpy as C

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def test_shapes() -> None:
    print("\n== 1. 前向形状 ==")
    model = C.SmallCNN(seed=0)
    x = np.random.default_rng(0).standard_normal((2, 1, 28, 28))
    shapes = []
    for layer in model.layers:
        x = layer.forward(x)
        shapes.append((type(layer).__name__, x.shape))
        print(f"   {type(layer).__name__:<10} -> {x.shape}")
    check("最后一层输出 10 类 logits", shapes[-1][1] == (2, 10), str(shapes[-1][1]))
    check("展平维度 = 16*5*5 = 400", model.flat_dim == 400, str(model.flat_dim))

    n_params = sum(p.size for _, p, _ in model.params())
    print(f"   参数量 = {n_params:,}")
    check("参数量合理（1e4 ~ 1e5）", 10_000 < n_params < 100_000, f"{n_params:,}")


def test_pool_backward_padding() -> None:
    print("\n== 2. 池化反向传播的边界处理 ==")
    pool = C.MaxPool2D(2)
    x = np.random.rand(2, 3, 11, 11)          # 奇数边长，最后一行/列会被丢弃
    y = pool.forward(x)
    dx = pool.backward(np.ones_like(y))
    check("前向输出 5x5（11//2）", y.shape == (2, 3, 5, 5), str(y.shape))
    check("反向还原成原始 11x11", dx.shape == x.shape, f"{dx.shape} vs {x.shape}")
    check("被丢弃的行列梯度为 0",
          np.allclose(dx[:, :, 10, :], 0) and np.allclose(dx[:, :, :, 10], 0))
    # 数值法验证池化反向：对若干随机位置做中心差分
    eps = 1e-6
    worst = 0.0
    rng = np.random.default_rng(0)
    for _ in range(6):
        n, c = rng.integers(0, 2), rng.integers(0, 3)
        i, j = rng.integers(0, 11, 2)
        x2 = x.copy()
        x2[n, c, i, j] += eps
        up = pool.forward(x2).sum()
        x2[n, c, i, j] -= 2 * eps
        dn = pool.forward(x2).sum()
        num = (up - dn) / (2 * eps)
        ana = dx[n, c, i, j]
        worst = max(worst, abs(num - ana))
    check("池化梯度与数值梯度一致（6 个随机位置）", worst < 1e-5, f"最大偏差={worst:.2e}")
    check("至少有一个被检查位置梯度非零（确保检查有效）", dx.sum() > 0, f"sum={dx.sum():.1f}")


def test_grad_check() -> None:
    print("\n== 3. 数值梯度校验（卷积/全连接权重）==")
    model = C.SmallCNN(seed=0)
    worst = C.grad_check(model, batch=3, num_checks=4, seed=1)
    check("最大相对误差 < 1e-4", worst < 1e-4, f"{worst:.3e}")


def test_overfit_batch() -> None:
    print("\n== 4. 单批过拟合测试 ==")
    rng = np.random.default_rng(3)
    ds = C.make_digit_dataset(n_train=10, n_test=10, seed=3, rasterize=False)
    model = C.SmallCNN(seed=3)
    x, y = ds.x_train[:10], ds.y_train[:10]
    mean, std = float(x.mean()), float(x.std()) + 1e-9
    x = (x - mean) / std
    opt = C.SGD(model, lr=0.02)
    for _ in range(120):
        probs = C.softmax(model.forward(x))
        model.backward(C.cross_entropy_grad(probs, y))
        opt.step(model)
    pred = model.predict(x)
    acc = float((pred == y).mean())
    check("小数据能被拟合到 100%", acc == 1.0, f"acc={acc:.2f}")


def test_learning() -> None:
    print("\n== 5. 短训练 vs 随机猜测 ==")
    ds = C.make_digit_dataset(n_train=800, n_test=200, seed=7, rasterize=True)
    C.normalize(ds)
    model = C.SmallCNN(seed=7)
    C.train(model, ds, epochs=4, batch_size=32, lr=3e-3, seed=7, verbose=False)
    acc = float((model.predict(ds.x_test) == ds.y_test).mean())
    check("测试准确率 > 0.5（远高于 0.1 随机）", acc > 0.5, f"acc={acc:.3f}")

    # 显著高于随机：用二项检验的粗略判据
    n = len(ds.y_test)
    se = (0.1 * 0.9 / n) ** 0.5
    check("显著优于随机猜测（> 随机 + 10 个标准差）", acc > 0.1 + 10 * se,
          f"acc={acc:.3f} 随机={0.1:.3f} se={se:.3f}")


def test_checkpoint_roundtrip() -> None:
    print("\n== 6. 权重保存 / 加载 ==")
    import os
    import shutil
    tmp = os.path.join("outputs", "_smoke_tmp")     # 写在 workspace 内，避免沙箱拒绝
    os.makedirs(tmp, exist_ok=True)
    try:
        model = C.SmallCNN(seed=11)
        x = np.random.default_rng(11).standard_normal((2, 1, 28, 28))
        before = model.forward(x)
        path = os.path.join(tmp, "m.npz")
        model.save_npz(path)
        loader = C.SmallCNN(seed=999)               # 换一组随机初始化的权重
        loader.load_npz(path)
        after = loader.forward(x)
        check("加载后输出完全一致", np.allclose(before, after, atol=1e-12))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 62)
    print("CNN 实现自检 (cnn_numpy.py)")
    print("=" * 62)
    test_shapes()
    test_pool_backward_padding()
    test_grad_check()
    test_overfit_batch()
    test_learning()
    test_checkpoint_roundtrip()

    print("\n" + "=" * 62)
    if FAILED:
        print(f"结果: {len(FAILED)} 项失败 -> {', '.join(FAILED)}")
        return 1
    print("结果: 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

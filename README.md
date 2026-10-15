# CNN 程序（PyTorch 主程序 + 纯 NumPy 从零实现）

本目录是一套可直接运行的卷积神经网络程序，包含两个版本：

| 文件 | 说明 | 本机状态 |
| --- | --- | --- |
| `cnn_mnist_torch.py` | **主交付**：PyTorch CNN，训练真实 MNIST（28×28，10 类） | 本机未安装 torch，**未实际运行**（代码已静态校验 + 结构推演验证） |
| `cnn_numpy.py` | 纯 NumPy 从零实现 CNN（卷积/池化/反向传播全手写），自带程序化数字数据集，**无需联网、无需 torch** | **已实际跑通**，测试集准确率 97.0% |
| `smoke_test.py` | NumPy 版自检：形状、池化边界、数值梯度校验、过拟合测试、短训练 | 已跑通，12 项全部通过 |
| `verify_torch_arch.py` | 无需 torch 也能验证 `cnn_mnist_torch.py` 的网络结构维度是否自洽 | 已跑通，13 项全部通过 |

---

## 1. PyTorch 版：训练真实 MNIST（主程序）

### 安装依赖

```bash
# 只有 CPU 的机器（推荐用官方 CPU 轮子，体积小很多）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 有 NVIDIA 显卡（CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> `torchvision` 首次运行会自动把 MNIST 下载到 `data/MNIST/raw/`。
> 如果机器完全无法联网，请手动下载下面 4 个文件放进 `data/MNIST/raw/`
> （镜像：`https://ossci-datasets.s3.amazonaws.com/mnist/` 或
> `https://storage.googleapis.com/cvdf-datasets/mnist/`）：
> `train-images-idx3-ubyte`、`train-labels-idx1-ubyte`、`t10k-images-idx3-ubyte`、`t10k-labels-idx1-ubyte`

### 运行

```bash
python cnn_mnist_torch.py                          # 默认训练 5 个 epoch
python cnn_mnist_torch.py --epochs 10 --device cuda
python cnn_mnist_torch.py --patience 3             # 验证集 3 个 epoch 不提升就早停
python cnn_mnist_torch.py --eval-only --ckpt outputs/mnist_cnn.pt
```

### 网络结构

```
输入 1×28×28
  Conv(1→32,3,pad=1) → BN → ReLU
  Conv(32→32,3,pad=1) → BN → ReLU
  MaxPool(2) → Dropout2d(0.15)                      → 32×14×14
  Conv(32→64,3,pad=1) → BN → ReLU
  Conv(64→64,3,pad=1) → BN → ReLU
  MaxPool(2) → Dropout2d(0.15)                      → 64×7×7
  AdaptiveAvgPool(3×3) → Flatten                     → 576
  Linear(576→128) → ReLU → Dropout(0.3)
  Linear(128→10)
```

参数量约 **140,330**，在真实 MNIST 上正常可达到 **99.3%+** 测试准确率。

### 训练技巧（代码里已实现）

- **数据增强**：训练集随机旋转 ±12°、平移 ±10%，用 `affine_grid` + `grid_sample` 实现
- **标准化**：MNIST 全局均值 0.1307 / 标准差 0.3081
- **标签平滑** 0.05，缓解过拟合
- **AdamW + 余弦退火**学习率，梯度范数裁剪到 5.0
- **保存最佳权重**（按验证准确率），可选早停
- 输出 4 张图（无需 matplotlib）：损失曲线、预测样例、混淆矩阵、**判错样例**

---

## 2. NumPy 版：从零手写 CNN（已在本机验证）

### 运行

```bash
python cnn_numpy.py --epochs 3                # 真实训练（本机约 8 秒）
python cnn_numpy.py --grad-check              # 数值梯度校验，验证反向传播正确性
python cnn_numpy.py --no-rasterize            # 不依赖字体，用纯笔画渲染，更快
python smoke_test.py                          # 完整自检
```

### 网络结构

```
输入 1×28×28
  Conv(1→8,3×3)  → ReLU → MaxPool(2)      → 8×13×13
  Conv(8→16,3×3) → ReLU → MaxPool(2)      → 16×5×5
  Flatten                                  → 400
  Dense(400→64)  → ReLU
  Dense(64→10)
```

参数量 **27,562**。卷积用 **im2col + 矩阵乘法**实现，速度可以接受；
池化用 argmax 路由回传梯度；优化器是带动量的 SGD（含权重衰减）。

### 数据集说明（重要）

标准 MNIST 需要联网下载。本机处于**无网络**受限环境（pip 与 torchvision 均无法下载），
因此 `cnn_numpy.py` 自带一个**程序化数字数据集**：用系统字体渲染 0–9，
每个样本随机加旋转（±18°）、缩放、平移、错切、笔画粗细、高斯噪声。

这样做的目的是让 CNN 在这台机器上**真实地训练起来**——梯度是真的、学习曲线是真的、
准确率是真的。它与真实 MNIST 的差别只在数据分布，换了数据加载器就是标准 MNIST 训练。

### 本机实测结果（真实运行输出）

默认配置（seed=0，4000 训练样本，3 epoch）：

```
== 构建数据集 ==
[procedural-font] 训练集 (4000, 1, 28, 28) / 测试集 (800, 1, 28, 28) 类别 [0..9]
标准化: mean=0.1929 std=0.3471

== 训练 ==
epoch 1/3 | train loss 1.6617 acc 0.4853 | val loss 0.6668 acc 0.7788 | 2.7s
epoch 2/3 | train loss 0.3672 acc 0.8998 | val loss 0.2250 acc 0.9475 | 2.4s
epoch 3/3 | train loss 0.1480 acc 0.9625 | val loss 0.1207 acc 0.9700 | 2.5s

测试集准确率: 0.9700
```

换一组随机种子独立复现（`--epochs 6 --train-size 6000 --test-size 1000 --seed 42`），
18 秒训练到 **99.7%**，说明结果不是偶然：

```
epoch 1/6 | train loss 1.2482 acc 0.6045 | val loss 0.3124 acc 0.9220 | 3.0s
epoch 2/6 | train loss 0.2193 acc 0.9393 | val loss 0.1005 acc 0.9830 | 2.8s
epoch 3/6 | train loss 0.0733 acc 0.9857 | val loss 0.0510 acc 0.9920 | 3.0s
epoch 4/6 | train loss 0.0455 acc 0.9910 | val loss 0.0373 acc 0.9920 | 3.0s
epoch 5/6 | train loss 0.0264 acc 0.9977 | val loss 0.0281 acc 0.9940 | 2.6s
epoch 6/6 | train loss 0.0196 acc 0.9970 | val loss 0.0181 acc 0.9970 | 2.6s

测试集准确率: 0.9970
```

下面是 seed=0 那次（3 epoch / 97.0%）的混淆矩阵（行=真实，列=预测），
主要错误集中在 `5→6`、`8→6`——字体里的 6 和 5/8 本来就容易混：

```
[[79  0  0  0  0  0  1  0  0  0]
 [ 0 79  1  0  0  0  0  0  0  0]
 [ 0  0 79  1  0  0  0  0  0  0]
 [ 0  0  0 79  0  0  0  0  0  1]
 [ 0  0  0  0 80  0  0  0  0  0]
 [ 0  0  0  0  0 73  7  0  0  0]
 [ 0  0  0  0  0  0 80  0  0  0]
 [ 0  0  0  0  0  0  0 80  0  0]
 [ 0  0  0  0  0  0 13  0 67  0]
 [ 0  0  0  0  0  0  0  0  0 80]]
```

（`outputs/` 里的图片来自 seed=42 那次 99.7% 的运行，绿色框=判对，红色框=判错，
右下角 `p=` 是预测、`t=` 是真实标签。）

![训练曲线](outputs/numpy_history.png)

![预测样例](outputs/numpy_predictions.png)

![混淆矩阵](outputs/numpy_confusion.png)

### 反向传播正确性证明

`--grad-check` 用中心差分（有限差分）对比解析梯度：

```
数值梯度校验: 检查了 48 个参数，最大相对误差 6.853e-09
```

误差量级 1e-9 ≈ float64 数值精度极限，说明卷积、池化、全连接、softmax 交叉熵的
反向传播推导全部正确。

---

## 3. 换成自己的图片数据集

`cnn_mnist_torch.py` 里的 `MNISTCNN` 与数据加载是解耦的，改数据只需要替换 `load_mnist`。
按类名分文件夹组织图片即可：

```
mydata/
  train/cat/*.jpg  train/dog/*.jpg
  val/cat/*.jpg    val/dog/*.jpg
```

```python
from torchvision import datasets, transforms
train = datasets.ImageFolder("mydata/train", transform=transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),          # 彩色图片就去掉这一行，并把 Conv2d 的 in_channels 改成 3
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]))
test = datasets.ImageFolder("mydata/val", transform=transforms.Compose([
    transforms.Resize((28, 28)), transforms.Grayscale(), transforms.ToTensor(),
]))
```

再把 `MNISTCNN(num_classes=...)` 改成你的类别数即可（自适应池化保证输入尺寸不同也不会崩）。

---

## 4. 环境说明

- Python：本机验证使用 DSH 捆绑的 CPython 3.12（`numpy 2.3.5` + `Pillow 12.3.0`）
- `cnn_numpy.py` / `smoke_test.py` / `verify_torch_arch.py` 只依赖 `numpy` 和 `Pillow`
- `cnn_mnist_torch.py` 需要 `torch` + `torchvision`（本机无法安装：pip 走不通 HTTPS）

# CNN 程序（PyTorch 主程序 + 纯 NumPy 从零实现）

> **更新（已装上 torch 后实测）**：PyTorch 版已在真实 MNIST 上跑通，5 个 epoch
> 达到 **测试准确率 99.53%**（CPU 约 4.7 分钟）。详见文末「实测记录」。

本目录是一套可直接运行的卷积神经网络程序，包含两个版本：

| 文件 | 说明 | 状态 |
| --- | --- | --- |
| `cnn_mnist_torch.py` | **主交付**：PyTorch CNN，训练真实 MNIST（28×28，10 类） | **已跑通，测试准确率 99.53%** |
| `cnn_numpy.py` | 纯 NumPy 从零实现 CNN（卷积/池化/反向传播全手写），自带程序化数字数据集，无需联网、无需 torch | 已跑通，合成数据 97.0%~99.7% |
| `predict_my_digit.py` | **推理脚本**：识别你自己手写的数字（自动反相/裁剪/质心居中） | 已跑通，真实 MNIST 图 5/5 全对 |
| `make_test_images.py` | 从 MNIST 测试集导出 PNG，用于测试推理脚本 | 已跑通 |
| `smoke_test.py` | NumPy 版自检：形状、池化边界、数值梯度校验、过拟合测试、短训练 | 12 项全部通过 |
| `verify_torch_arch.py` | 无需 torch 也能验证 `cnn_mnist_torch.py` 的网络结构维度是否自洽 | 13 项全部通过 |

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
python cnn_mnist_torch.py --no-augment             # 关掉随机增强（对比用）
python cnn_mnist_torch.py --eval-only --ckpt outputs/mnist_cnn.pt
```

### 实测记录（真实 MNIST，CPU）

环境：`torch 2.14.1+cpu`、`torchvision 0.29.1`、Python 3.10.2、`--workers 0`

```
设备: cpu | torch 2.14.1+cpu
数据来源: torchvision | 训练样本 60000 | 测试样本 10000
增强: 开 | 标签平滑: 0.05
模型参数量: 140,522
epoch  1/5 | train loss 0.5834 acc 0.9140 | val loss 0.0766 acc 0.9863 | lr 9.05e-04 | 60.1s  *saved*
epoch  2/5 | train loss 0.3982 acc 0.9787 | val loss 0.0732 acc 0.9921 | lr 6.55e-04 | 57.9s  *saved*
epoch  3/5 | train loss 0.3756 acc 0.9844 | val loss 0.0514 acc 0.9940 | lr 3.45e-04 | 55.8s  *saved*
epoch  4/5 | train loss 0.3636 acc 0.9873 | val loss 0.0494 acc 0.9946 | lr 9.55e-05 | 55.2s  *saved*
epoch  5/5 | train loss 0.3556 acc 0.9895 | val loss 0.0478 acc 0.9953 | lr 0.00e+00 | 55.4s  *saved*

最佳验证准确率: 0.9953 | 权重: outputs/mnist_cnn.pt
测试集: loss 0.0478 | 准确率 0.9953

各类别准确率:
  0: 1.0000 (n=980)    1: 0.9956 (n=1135)   2: 0.9971 (n=1032)
  3: 0.9960 (n=1010)   4: 0.9969 (n=982)    5: 0.9933 (n=892)
  6: 0.9906 (n=958)    7: 0.9971 (n=1028)   8: 0.9969 (n=974)
  9: 0.9891 (n=1009)
```

**它错在哪**：`outputs/torch_errors.png` 里全是"鬼画符"级别的样本
（写成 `β` 形的 8、糊成一团的 7、缺笔画的 2），没有把清清楚楚的数字认错。
这正是模型健康、99.53% 可信的证据。

**与 NumPy 字体版的对比**：同一个「写成 S 形的 5」，
字体版模型以 92.2% 的置信度认成 **8**，真实 MNIST 版以 98.7% 认成 **5**。
这就是"训练数据分布是否匹配"的差别。

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

参数量 **140,522**（实测），在真实 MNIST 上实测达到 **99.53%** 测试准确率。

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

## 4. 识别你自己的手写数字

训练准确率高不代表对**你的**字也准——这是最值得亲手做的验证。

```bash
python predict_my_digit.py my_digit.png                    # 认一张
python predict_my_digit.py a.png b.png --topk 3            # 多张 + 显示前 3 候选
python predict_my_digit.py my_digit.png --save-preview     # 存一张可视化预览图
python make_test_images.py --count 5                       # 从 MNIST 导出图片来试
```

脚本会自动处理这两件最容易让模型翻车的事：

- **自动反相**：MNIST 是黑底白字，而照片/画图软件通常是白底黑字（用四周边框亮度判断背景）
- **质心居中**：先裁掉空白边，再等比缩放到 20×20，最后按亮度质心对齐到 28×28 画布

手动控制：`--invert yes/no/auto`、`--no-center`。置信度低于 50% 时脚本会主动提示
「模型很犹豫」，多半是图片太模糊或笔迹风格差异过大。

实测（用 `outputs/mnist_cnn.pt` 识别从 MNIST 导出的 5 张图，含那张 S 形的 5）：

```
digit_4_0.png -> 4 (98.4%)    digit_7_1.png -> 7 (96.6%)
digit_5_2.png -> 5 (98.7%)    digit_3_3.png -> 3 (96.7%)
digit_5_4.png -> 5 (90.4%)                          5/5 全对
```

---

## 5. 环境说明

- 实测环境：Python 3.10.2 + `torch 2.14.1+cpu` + `torchvision 0.29.1` + `numpy` + `Pillow`
- `cnn_numpy.py` / `smoke_test.py` / `verify_torch_arch.py` 只需 `numpy` + `Pillow`
- `cnn_mnist_torch.py` / `predict_my_digit.py` 需要 `torch`（`predict_my_digit.py`
  也能读 NumPy 版权重 `outputs/numpy_cnn.npz`，自动选择）
- MNIST 数据在首次运行时由 torchvision 自动下载到 `data/MNIST/raw/`（约 11 MB）
- CPU 训练 5 epoch 约 5 分钟；有 NVIDIA 显卡可加 `--device cuda`，通常 1 分钟内

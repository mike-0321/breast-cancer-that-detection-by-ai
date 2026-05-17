# MONAI DenseNet121 本地运行说明（PyCharm）

## 1) 准备环境
建议 Python 3.9 ~ 3.11。

在项目根目录（computer science studio2）打开终端后执行：

```bash
pip install -r requirements.txt
```

如果你有 NVIDIA GPU，可按 PyTorch 官网安装对应 CUDA 版本的 torch。

---

## 2) 数据集结构
当前脚本默认读取 `archive/`，目录必须是：

```text
archive/
  train/
    0/
    1/
  valid/
    0/
    1/
  test/
    0/
    1/
```

你现在的目录已经是这个结构，可以直接跑。

---

## 3) 在 PyCharm 运行
运行文件：

- `monai_DenseNet121_2_local.py`

默认参数下直接运行即可。

你也可以在 Run Configuration 的 Parameters 里自定义：

```bash
--data_dir archive --epochs 5 --batch_size 16 --num_workers 0 --lr 0.0001
```

> Mac / PyCharm 下，`num_workers=0` 更稳，避免多进程读取问题。

---

## 4) 输出结果
运行后会在当前目录生成：

- `densenet121_monai.pth`（模型权重）
- `training_loss.png`
- `validation_accuracy.png`

终端会输出 Train/Valid/Test 数量、每轮 loss、验证准确率和最终测试准确率。

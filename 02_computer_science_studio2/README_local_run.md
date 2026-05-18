# MONAI DenseNet121 Local Run Guide

This note explains how to run the local MONAI DenseNet121 script in PyCharm or from a terminal.

## 1. Environment

Use Python 3.9 to 3.11.

From the `02_computer_science_studio2/` folder, install the required packages:

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the PyTorch version that matches your CUDA version from the official PyTorch installation guide.

## 2. Dataset Layout

The local script expects the dataset folder to be named `archive/` and arranged like this:

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

The labels use `0` and `1` folders for the binary classification task.

## 3. Run In PyCharm

Run this file:

```text
monai_DenseNet121_2_local.py
```

The default settings can be used directly. To customise the run, add parameters in the PyCharm Run Configuration:

```bash
--data_dir archive --epochs 5 --batch_size 16 --num_workers 0 --lr 0.0001
```

On macOS or PyCharm, `num_workers=0` is usually more stable because it avoids multiprocessing issues during image loading.

## 4. Outputs

The script writes the following files to the current folder:

- `densenet121_monai.pth`
- `training_loss.png`
- `validation_accuracy.png`

The terminal output reports the Train/Valid/Test counts, epoch loss, validation accuracy, and final test accuracy.

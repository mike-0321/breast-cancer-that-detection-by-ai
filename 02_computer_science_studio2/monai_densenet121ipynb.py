# MONAI DenseNet121 notebook script

# Google Drive mount for Colab runs.
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ModuleNotFoundError:
    print("Google Colab is not available. For local runs, use monai_DenseNet121_2_local.py.")

# In Colab, run this command before the imports if MONAI is not installed:
# !pip install -q monai

# Imports

import os
import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    EnsureType,
    ScaleIntensity,
    Resize,
    RepeatChannel,
    RandFlip,
    RandRotate90,
    RandZoom
)
from monai.data import ImageDataset, DataLoader
from monai.networks.nets import DenseNet121

# Path settings

base_dir = "/content/drive/MyDrive/breast cancer"

train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir  = os.path.join(base_dir, "test")

# Build image paths and labels from class folders.

def build_file_list(data_dir):
    image_files = []
    labels = []

    for class_name in ["0", "1"]:
        class_dir = os.path.join(data_dir, class_name)
        files = glob.glob(os.path.join(class_dir, "*"))
        image_files.extend(files)
        labels.extend([int(class_name)] * len(files))

    return image_files, labels

train_files, train_labels = build_file_list(train_dir)
valid_files, valid_labels = build_file_list(valid_dir)
test_files, test_labels   = build_file_list(test_dir)

print("Train:", len(train_files))
print("Valid:", len(valid_files))
print("Test :", len(test_files))

# MONAI preprocessing

train_transforms = Compose([
    # LoadImage(image_only=True), # This line caused the error, removing it
    EnsureChannelFirst(),
    EnsureType(),
    ScaleIntensity(),
    Resize((224, 224)),
    RandFlip(prob=0.5, spatial_axis=1),
    RandRotate90(prob=0.5),
    RandZoom(prob=0.3, min_zoom=0.9, max_zoom=1.1),
])

val_transforms = Compose([
    # LoadImage(image_only=True), # This line caused the error, removing it
    EnsureChannelFirst(),
    EnsureType(),
    ScaleIntensity(),
    Resize((224, 224)),
])

# Dataset and dataloader setup

train_ds = ImageDataset(
    image_files=train_files,
    labels=train_labels,
    transform=train_transforms
)

valid_ds = ImageDataset(
    image_files=valid_files,
    labels=valid_labels,
    transform=val_transforms
)

test_ds = ImageDataset(
    image_files=test_files,
    labels=test_labels,
    transform=val_transforms
)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

print(len(train_ds))
print(train_ds[0])

# Inspect one preprocessed batch.

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Min pixel:", images.min().item())
print("Max pixel:", images.max().item())

# Display two preprocessed images.

plt.figure(figsize=(8, 4))

for i in range(2):
    plt.subplot(1, 2, i + 1)
    # Select only the first 3 channels for display, as imshow expects 3 or 4 channels
    img = images[i][:3].permute(1, 2, 0).numpy()   # C,H,W -> H,W,C, taking only the first 3 channels
    plt.imshow(img)
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# DenseNet121 model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenseNet121(
    spatial_dims=2,
    in_channels=3,
    out_channels=2
).to(device)

print(model)

# Loss function and optimizer

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop

max_epochs = 5

train_losses = []
val_accuracies = []

for epoch in range(max_epochs):
    print("-" * 30)
    print(f"Epoch {epoch + 1}/{max_epochs}")

    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        step += 1

    epoch_loss /= step
    train_losses.append(epoch_loss)
    print(f"Training loss: {epoch_loss:.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for val_data in valid_loader:
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(val_images)
            preds = torch.argmax(val_outputs, dim=1)

            correct += torch.sum(preds == val_labels).item()
            total += len(val_labels)

    val_acc = correct / total
    val_accuracies.append(val_acc)
    print(f"Validation accuracy: {val_acc:.4f}")

# Test-set evaluation

model.eval()
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels_batch = test_data
        test_images = test_images.to(device)
        test_labels_batch = test_labels_batch.to(device)

        test_outputs = model(test_images)
        preds = torch.argmax(test_outputs, dim=1)

        correct += torch.sum(preds == test_labels_batch).item()
        total += len(test_labels_batch)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(test_labels_batch.cpu().numpy())

test_acc = correct / total
print(f"Test accuracy: {test_acc:.4f}")

# Training result plots

plt.figure(figsize=(6, 4))
plt.plot(range(1, max_epochs + 1), train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(range(1, max_epochs + 1), val_accuracies, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Curve")
plt.show()

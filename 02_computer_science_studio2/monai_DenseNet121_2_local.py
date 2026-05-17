import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from monai.data import DataLoader, ImageDataset
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    RandFlip,
    RandRotate90,
    RandZoom,
    Resize,
    ScaleIntensity,
)


def build_file_list(data_dir: Path):
    image_files = []
    labels = []

    for class_name in ["0", "1"]:
        class_dir = data_dir / class_name
        files = sorted(glob.glob(str(class_dir / "*")))
        image_files.extend(files)
        labels.extend([int(class_name)] * len(files))

    return image_files, labels


def check_dataset_structure(base_dir: Path):
    required = [
        base_dir / "train" / "0",
        base_dir / "train" / "1",
        base_dir / "valid" / "0",
        base_dir / "valid" / "1",
        base_dir / "test" / "0",
        base_dir / "test" / "1",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset structure is incomplete. Missing paths:\n- " + "\n- ".join(missing)
        )


def main():
    parser = argparse.ArgumentParser(description="Train MONAI DenseNet121 locally")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="archive",
        help="Dataset root containing train/ valid/ test folders",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_model", type=str, default="densenet121_monai.pth")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = (script_dir / args.data_dir).resolve()

    print(f"Using dataset: {base_dir}")
    check_dataset_structure(base_dir)

    train_dir = base_dir / "train"
    valid_dir = base_dir / "valid"
    test_dir = base_dir / "test"

    train_files, train_labels = build_file_list(train_dir)
    valid_files, valid_labels = build_file_list(valid_dir)
    test_files, test_labels = build_file_list(test_dir)

    print("Train:", len(train_files))
    print("Valid:", len(valid_files))
    print("Test :", len(test_files))

    train_transforms = Compose(
        [
            EnsureChannelFirst(),
            EnsureType(),
            ScaleIntensity(),
            Resize((224, 224)),
            RandFlip(prob=0.5, spatial_axis=1),
            RandRotate90(prob=0.5),
            RandZoom(prob=0.3, min_zoom=0.9, max_zoom=1.1),
        ]
    )

    val_transforms = Compose(
        [
            EnsureChannelFirst(),
            EnsureType(),
            ScaleIntensity(),
            Resize((224, 224)),
        ]
    )

    train_ds = ImageDataset(image_files=train_files, labels=train_labels, transform=train_transforms)
    valid_ds = ImageDataset(image_files=valid_files, labels=valid_labels, transform=val_transforms)
    test_ds = ImageDataset(image_files=test_files, labels=test_labels, transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        print("-" * 30)
        print(f"Epoch {epoch + 1}/{args.epochs}")

        model.train()
        epoch_loss = 0.0
        step = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        epoch_loss = epoch_loss / max(step, 1)
        train_losses.append(epoch_loss)
        print(f"Training loss: {epoch_loss:.4f}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for val_images, val_labels in valid_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                preds = torch.argmax(val_outputs, dim=1)

                correct += torch.sum(preds == val_labels).item()
                total += len(val_labels)

        val_acc = correct / max(total, 1)
        val_accuracies.append(val_acc)
        print(f"Validation accuracy: {val_acc:.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for test_images, test_labels_batch in test_loader:
            test_images = test_images.to(device)
            test_labels_batch = test_labels_batch.to(device)

            test_outputs = model(test_images)
            preds = torch.argmax(test_outputs, dim=1)

            correct += torch.sum(preds == test_labels_batch).item()
            total += len(test_labels_batch)

    test_acc = correct / max(total, 1)
    print(f"Test accuracy: {test_acc:.4f}")

    save_path = (script_dir / args.save_model).resolve()
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, args.epochs + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(script_dir / "training_loss.png", dpi=150)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, args.epochs + 1), val_accuracies, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.tight_layout()
    plt.savefig(script_dir / "validation_accuracy.png", dpi=150)

    print("Saved plots: training_loss.png, validation_accuracy.png")


if __name__ == "__main__":
    main()

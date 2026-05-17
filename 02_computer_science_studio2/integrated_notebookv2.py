# ============================================================================
# Cell 1: Mount Google Drive
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# Cell 2: Install Dependencies
# ============================================================================
# !pip install -q monai scikit-learn pydicom opencv-python-headless

# ============================================================================
# Cell 3: Import Libraries
# ============================================================================
import os, glob, shutil, cv2, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from monai.transforms import (
    Compose, EnsureType,
    RandFlip, RandRotate90, RandZoom,
    RandGaussianNoise, RandAdjustContrast
)
from monai.networks.nets import DenseNet121
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ============================================================================
# Cell 4: Path Configuration
# ============================================================================
base_dir        = '/content/drive/MyDrive/breast cancer'
train_dir       = os.path.join(base_dir, 'train')
valid_dir       = os.path.join(base_dir, 'valid')
test_dir        = os.path.join(base_dir, 'test')
processed_dir   = os.path.join(base_dir, 'processed_v2')
processed_train = os.path.join(processed_dir, 'train')
processed_valid = os.path.join(processed_dir, 'valid')
processed_test  = os.path.join(processed_dir, 'test')
model_path      = '/content/drive/MyDrive/best_densenet121_v2.pth'
IMG_SIZE        = 224


# ============================================================================
# Cell 5: Image Loading
# ============================================================================

def load_image_grayscale(file_path):
    """Load any image as grayscale float32, range [0, 1]."""
    try:
        if file_path.lower().endswith('.dcm'):
            import pydicom
            dcm = pydicom.dcmread(file_path)
            img = dcm.pixel_array.astype(np.float32)
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                img = img.max() - img
        else:
            img = np.array(Image.open(file_path).convert('L'), dtype=np.float32)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    except Exception as e:
        print(f"Load error {file_path}: {e}")
        return None


# ============================================================================
# Cell 6: White-BG Detection
# ============================================================================

def is_white_background(image, wr_thresh=0.3, br_thresh=0.5, mean_thresh=0.7):
    """Check if image has a white background (returns True if white)."""
    wr = (image > 0.95).sum() / image.size
    br = (image > 0.80).sum() / image.size
    if wr > wr_thresh or br > br_thresh:
        return True
    if image.mean() > mean_thresh and image.std() < 0.2:
        return True
    return False


# ============================================================================
# Cell 7: Priority 1 — Breast Mask + Region Crop (NO stretching)
# ============================================================================

def create_breast_mask(image):
    """
    Create a binary breast mask using Otsu thresholding + morphological ops.
    Returns: binary mask (H, W), same size as input, 1 = breast, 0 = background.
    """
    img_uint8 = (image * 255).astype(np.uint8)

    # Otsu threshold
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning: close small holes, open small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Keep only the largest connected component (= breast)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return (binary > 0).astype(np.uint8)

    # Component 0 is background, find largest among the rest
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8)
    return mask


# ============================================================================
# Cell 8: Priority 2 — Pectoral Muscle Removal (simple approach)
# ============================================================================

def remove_pectoral_muscle(image, mask):
    """
    Approximate pectoral muscle removal for MLO views.
    Detects the bright triangular region in the upper corner and masks it out.
    Returns: updated mask with pectoral region set to 0.
    """
    h, w = image.shape
    roi_h, roi_w = h // 3, w // 3

    # Determine breast side: left or right
    left_sum  = mask[:, :w // 2].sum()
    right_sum = mask[:, w // 2:].sum()

    if left_sum >= right_sum:
        # Breast on left side -> pectoral in upper-left
        roi = image[:roi_h, :roi_w]
        roi_mask = mask[:roi_h, :roi_w]
    else:
        # Breast on right side -> pectoral in upper-right
        roi = image[:roi_h, w - roi_w:]
        roi_mask = mask[:roi_h, w - roi_w:]

    # Threshold bright region in ROI (pectoral is usually brightest tissue)
    if roi_mask.sum() == 0:
        return mask

    breast_pixels = roi[roi_mask > 0]
    if len(breast_pixels) == 0:
        return mask

    pec_thresh = np.percentile(breast_pixels, 90)
    pec_mask = ((roi > pec_thresh) & (roi_mask > 0)).astype(np.uint8)

    # Morphological closing to connect pectoral region
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    pec_mask = cv2.morphologyEx(pec_mask, cv2.MORPH_CLOSE, k, iterations=3)

    # Remove pectoral from breast mask
    updated_mask = mask.copy()
    if left_sum >= right_sum:
        updated_mask[:roi_h, :roi_w] = np.where(pec_mask > 0, 0, updated_mask[:roi_h, :roi_w])
    else:
        updated_mask[:roi_h, w - roi_w:] = np.where(pec_mask > 0, 0, updated_mask[:roi_h, w - roi_w:])

    return updated_mask


# ============================================================================
# Cell 9: Crop + Pad (NO stretching, aspect ratio preserved)
# ============================================================================

def crop_and_pad_breast(image, mask, target_size=224):
    """
    Crop breast region using mask bounding box, then pad to square
    (preserves aspect ratio, NO stretching), then resize.

    Returns:
        processed: float32 array (H, W), single channel, range [0, 1]
        mask_resized: resized breast mask (H, W), for mask-guided attention
    """
    # Apply mask to image (zero out background)
    masked_image = image * mask

    # Find bounding box of breast
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        # No breast found, return padded original
        masked_image = image
        h, w = image.shape
        r0, c0, r1, c1 = 0, 0, h, w
    else:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1

    # Crop
    cropped_img  = masked_image[r0:r1, c0:c1]
    cropped_mask = mask[r0:r1, c0:c1]

    # Pad to square (add padding to shorter side)
    ch, cw = cropped_img.shape
    max_side = max(ch, cw)
    pad_h = (max_side - ch) // 2
    pad_w = (max_side - cw) // 2

    padded_img = np.zeros((max_side, max_side), dtype=np.float32)
    padded_img[pad_h:pad_h + ch, pad_w:pad_w + cw] = cropped_img

    padded_mask = np.zeros((max_side, max_side), dtype=np.uint8)
    padded_mask[pad_h:pad_h + ch, pad_w:pad_w + cw] = cropped_mask

    # Resize to target (aspect ratio already preserved by padding)
    processed = cv2.resize(padded_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(padded_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    return processed, mask_resized


# ============================================================================
# Cell 10: Full Preprocessing Pipeline
# ============================================================================

def preprocess_mammogram(image, target_size=224, remove_pectoral=True):
    """
    Complete preprocessing:
      1. Create breast mask (Otsu + largest component)
      2. Remove pectoral muscle (optional)
      3. Crop to breast bounding box
      4. Pad to square (no stretching)
      5. Resize to target_size
      6. Keep single channel (no 3-ch expansion)

    Returns:
        processed_image: (target_size, target_size) float32
        processed_mask:  (target_size, target_size) uint8
    """
    mask = create_breast_mask(image)

    if remove_pectoral:
        mask = remove_pectoral_muscle(image, mask)

    processed_img, processed_mask = crop_and_pad_breast(image, mask, target_size)
    return processed_img, processed_mask


# ============================================================================
# Cell 11: Step 1 — Scan + Filter White-BG + Preprocess + Save
# ============================================================================

def preprocess_and_save_all(src_dir, dst_dir, target_size=224, remove_pectoral=True):
    """
    Scan source directory, skip white-BG images, preprocess the rest,
    save both processed image and breast mask as .npz files.
    """
    total = {'processed': 0, 'white_skipped': 0, 'failed': 0, 'exist': 0}

    for cls in ['0', '1']:
        src_cls = os.path.join(src_dir, cls)
        dst_cls = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)

        files = glob.glob(os.path.join(src_cls, '*'))
        split_name = os.path.basename(src_dir)
        print(f"\n  {split_name}/class_{cls}: {len(files)} files")

        for f in tqdm(files, desc=f"Class {cls}"):
            fname = os.path.splitext(os.path.basename(f))[0]
            save_path = os.path.join(dst_cls, f"{fname}.npz")

            if os.path.exists(save_path):
                total['exist'] += 1
                continue

            image = load_image_grayscale(f)
            if image is None:
                total['failed'] += 1
                continue

            # White-BG filter
            if is_white_background(image):
                total['white_skipped'] += 1
                continue

            # Full preprocessing
            proc_img, proc_mask = preprocess_mammogram(image, target_size, remove_pectoral)

            # Save image + mask together (single channel, no 3-ch expansion)
            np.savez_compressed(save_path, image=proc_img, mask=proc_mask)
            total['processed'] += 1

    print(f"\n  Done: processed={total['processed']}, exist={total['exist']}, "
          f"white_skipped={total['white_skipped']}, failed={total['failed']}")
    return total


# Clear old data if exists
if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
    print("Cleared old preprocessed data.\n")

print("=" * 70)
print("Step 1: Preprocess all splits (white-BG filter + breast crop + pectoral removal)")
print("=" * 70)

stats_all = {}
for split_name, src, dst in [('train', train_dir, processed_train),
                              ('valid', valid_dir, processed_valid),
                              ('test',  test_dir,  processed_test)]:
    print(f"\n--- {split_name.upper()} ---")
    stats_all[split_name] = preprocess_and_save_all(src, dst, IMG_SIZE, remove_pectoral=True)


# ============================================================================
# Cell 12: Visualize Preprocessing Results
# ============================================================================

def visualize_pipeline(src_dir, dst_dir, cls='1', n=4):
    """Show original → mask → cropped+padded for a few samples."""
    src_files = glob.glob(os.path.join(src_dir, cls, '*'))
    pairs = []
    for sf in src_files:
        fname = os.path.splitext(os.path.basename(sf))[0]
        npz = os.path.join(dst_dir, cls, f"{fname}.npz")
        if os.path.exists(npz):
            pairs.append((sf, npz))
        if len(pairs) >= n:
            break

    fig, axes = plt.subplots(len(pairs), 4, figsize=(16, 4 * len(pairs)))
    if len(pairs) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Preprocessing Pipeline: Original → Mask → Masked → Cropped+Padded',
                 fontsize=13, fontweight='bold')

    for i, (sf, npz_path) in enumerate(pairs):
        orig = load_image_grayscale(sf)
        data = np.load(npz_path)
        proc_img, proc_mask = data['image'], data['mask']

        # Also show intermediate mask on original
        orig_mask = create_breast_mask(orig)
        orig_masked = orig * orig_mask

        axes[i, 0].imshow(orig, cmap='gray'); axes[i, 0].set_title(f'Original\n{orig.shape}'); axes[i, 0].axis('off')
        axes[i, 1].imshow(orig_mask, cmap='gray'); axes[i, 1].set_title('Breast Mask'); axes[i, 1].axis('off')
        axes[i, 2].imshow(orig_masked, cmap='gray'); axes[i, 2].set_title('Masked Image'); axes[i, 2].axis('off')
        axes[i, 3].imshow(proc_img, cmap='gray'); axes[i, 3].set_title(f'Cropped+Padded\n{proc_img.shape}'); axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'preprocessing_pipeline_v2.png'), dpi=150, bbox_inches='tight')
    plt.show()

visualize_pipeline(train_dir, processed_train, cls='1', n=4)
visualize_pipeline(train_dir, processed_train, cls='0', n=4)


# ============================================================================
# Cell 13: Build File Lists + Dataset
# ============================================================================

def build_file_list(data_dir):
    files, labels = [], []
    for cls in ['0', '1']:
        cls_files = glob.glob(os.path.join(data_dir, cls, '*.npz'))
        files.extend(cls_files)
        labels.extend([int(cls)] * len(cls_files))
    return files, labels

train_files, train_labels = build_file_list(processed_train)
valid_files, valid_labels = build_file_list(processed_valid)
test_files,  test_labels  = build_file_list(processed_test)

print(f'After filtering:')
print(f'  Train: {len(train_files)} (c0={train_labels.count(0)}, c1={train_labels.count(1)})')
print(f'  Valid: {len(valid_files)} | Test: {len(test_files)}')


class MammogramDataset(Dataset):
    """
    Loads preprocessed .npz files containing:
      - 'image': single-channel grayscale (H, W)
      - 'mask': breast mask (H, W)
    Outputs image as (1, H, W) tensor (single channel, no expansion to 3-ch).
    """
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = data['image'].astype(np.float32)  # (H, W)
        mask  = data['mask'].astype(np.float32)    # (H, W)
        label = self.labels[idx]

        # Add channel dim: (1, H, W)
        image = image[np.newaxis, ...]
        mask  = mask[np.newaxis, ...]

        image = torch.from_numpy(image)
        mask  = torch.from_numpy(mask)

        if self.transform is not None:
            image = self.transform(image)

        return image, mask, label


# ============================================================================
# Cell 14: Augmentation + DataLoader
# ============================================================================

train_transforms = Compose([
    EnsureType(),
    RandFlip(prob=0.5, spatial_axis=0),
    RandFlip(prob=0.5, spatial_axis=1),
    RandRotate90(prob=0.5),
    RandZoom(prob=0.3, min_zoom=0.85, max_zoom=1.15),
    RandGaussianNoise(prob=0.2, mean=0.0, std=0.05),
    RandAdjustContrast(prob=0.2, gamma=(0.8, 1.2)),
])
val_transforms = Compose([EnsureType()])

BATCH_SIZE  = 16
NUM_WORKERS = 2

train_ds = MammogramDataset(train_files, train_labels, transform=train_transforms)
valid_ds = MammogramDataset(valid_files, valid_labels, transform=val_transforms)
test_ds  = MammogramDataset(test_files,  test_labels,  transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

img, msk, lbl = train_ds[0]
print(f'Sample: image={img.shape}, mask={msk.shape}, label={lbl}')


# ============================================================================
# Cell 15: Attention Modules
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class MaskGuidedSpatialAttention(nn.Module):
    """
    Spatial attention guided by breast mask.
    Learns where to attend within the breast region while suppressing
    background / pectoral muscle artifacts.

    The breast mask is downsampled to match feature map resolution,
    then multiplied into the spatial attention weights so the model
    can ONLY attend inside the breast.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        """
        Args:
            x:    feature map (B, C, H', W')
            mask: breast mask (B, 1, H, W), will be downsampled to (B, 1, H', W')
        """
        spatial_att = self.conv(x)  # (B, 1, H', W')

        if mask is not None:
            # Downsample mask to feature map resolution
            mask_down = F.interpolate(mask, size=spatial_att.shape[2:],
                                      mode='nearest')
            spatial_att = spatial_att * mask_down  # Zero out attention outside breast

        return x * spatial_att


class MedicalAttentionBlock(nn.Module):
    """
    Combined attention for medical imaging:
      1. SE channel attention (which feature channels matter)
      2. Mask-guided spatial attention (WHERE in the breast to look)

    This is more appropriate than generic CBAM for mammography because
    it explicitly uses the breast mask to prevent the model from learning
    background artifacts.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.se = SEBlock(in_channels, reduction)
        self.spatial = MaskGuidedSpatialAttention(in_channels)

    def forward(self, x, mask=None):
        x = self.se(x)
        x = self.spatial(x, mask)
        return x


# ============================================================================
# Cell 16: Model Definition
# ============================================================================

class DenseNetMedicalAttention(nn.Module):
    """
    DenseNet121 (single-channel input) + Mask-Guided Medical Attention.

    Key differences from v1:
      - in_channels=1 (grayscale, no artificial 3-ch expansion)
      - Attention is guided by breast mask (not generic CBAM)
      - All ReLU use inplace=False for Grad-CAM compatibility
    """
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super().__init__()
        backbone = DenseNet121(spatial_dims=2, in_channels=1, out_channels=1024)
        self.features = backbone.features
        self.attention = MedicalAttentionBlock(in_channels=1024, reduction=16)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, mask=None):
        feat = self.features(x)
        feat = self.attention(feat, mask)
        return self.classifier(feat)


model = DenseNetMedicalAttention(num_classes=2, dropout_rate=0.4).to(device)

# Disable inplace ReLU in backbone for Grad-CAM compatibility
def disable_inplace_relu(m):
    for child in m.children():
        if isinstance(child, nn.ReLU):
            child.inplace = False
        else:
            disable_inplace_relu(child)
disable_inplace_relu(model)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================================
# Cell 17: Training Configuration
# ============================================================================
MAX_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 10

n0 = train_labels.count(0)
n1 = train_labels.count(1)
w0 = (n0 + n1) / (2 * n0)
w1 = (n0 + n1) / (2 * n1)
class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
print(f'Class weights -> 0: {w0:.4f}, 1: {w1:.4f}')

loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)


# ============================================================================
# Cell 18: Training Loop (passes mask to model)
# ============================================================================
train_losses, val_accs, val_losses, lr_history = [], [], [], []
best_val_acc = 0.0
patience_count = 0

for epoch in range(MAX_EPOCHS):
    # --- Train ---
    model.train()
    epoch_loss, steps = 0.0, 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{MAX_EPOCHS}]")
    for imgs, masks, labels in loop:
        imgs   = imgs.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, masks)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        steps += 1
        loop.set_postfix(avg_loss=epoch_loss / steps)

    avg_train_loss = epoch_loss / steps
    train_losses.append(avg_train_loss)
    lr_history.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

    # --- Validate ---
    model.eval()
    correct, total_val, val_loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for imgs, masks, labels in valid_loader:
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            outputs = model(imgs, masks)
            val_loss_sum += loss_fn(outputs, labels).item()
            correct   += (torch.argmax(outputs, 1) == labels).sum().item()
            total_val += len(labels)

    val_acc      = correct / total_val
    avg_val_loss = val_loss_sum / len(valid_loader)
    val_accs.append(val_acc)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1:02d}/{MAX_EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | LR: {lr_history[-1]:.2e}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f"  Best model saved (Val Acc: {best_val_acc:.4f})")
        patience_count = 0
    else:
        patience_count += 1

    if patience_count >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")


# ============================================================================
# Cell 19: Training Curves
# ============================================================================
x = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(x, train_losses, 'o-', label='Train'); axes[0].plot(x, val_losses, 's-', label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
axes[1].plot(x, val_accs, 'o-'); axes[1].axhline(best_val_acc, ls='--', label=f'Best={best_val_acc:.4f}')
axes[1].set_title('Val Accuracy'); axes[1].legend(); axes[1].grid(True)
axes[2].plot(x, lr_history, 'o-'); axes[2].set_title('LR'); axes[2].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'training_curve_v2.png'), dpi=150)
plt.show()


# ============================================================================
# Cell 20: Evaluation
# ============================================================================
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_preds, all_labels_eval, all_probs = [], [], []
with torch.no_grad():
    for imgs, masks, labels in valid_loader:
        outputs = model(imgs.to(device), masks.to(device))
        all_probs.extend(F.softmax(outputs, 1)[:, 1].cpu().numpy())
        all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        all_labels_eval.extend(labels.numpy())

all_preds       = np.array(all_preds)
all_labels_eval = np.array(all_labels_eval)
all_probs       = np.array(all_probs)


# ============================================================================
# Cell 21: Confusion Matrix + Report
# ============================================================================
CLASS_NAMES = ['Benign', 'Malignant']
cm = confusion_matrix(all_labels_eval, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor='black')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix (DenseNet121 + Mask-Guided Attention)')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'confusion_matrix_v2.png'), dpi=150, bbox_inches='tight')
plt.show()

acc  = accuracy_score(all_labels_eval, all_preds)
prec = precision_score(all_labels_eval, all_preds, average='binary')
rec  = recall_score(all_labels_eval, all_preds, average='binary')
f1   = f1_score(all_labels_eval, all_preds, average='binary')
try:    auc = roc_auc_score(all_labels_eval, all_probs)
except: auc = None

print("\n===== Classification Metrics =====")
print(f"Accuracy : {acc:.4f}\nPrecision: {prec:.4f}\nRecall   : {rec:.4f}\nF1-score : {f1:.4f}")
if auc: print(f"AUC      : {auc:.4f}")
print("\n" + classification_report(all_labels_eval, all_preds, target_names=CLASS_NAMES, digits=4))


# ============================================================================
# Cell 22: Grad-CAM for Mask-Guided Attention Model
# ============================================================================

class GradCAM:
    """Grad-CAM adapted for mask-guided model (forward takes image + mask)."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, image_tensor, mask_tensor, target_class=None):
        self.model.eval()
        output = self.model(image_tensor, mask_tensor)
        probs  = torch.softmax(output, 1)
        if target_class is None:
            target_class = output.argmax(1).item()
        pred_prob = probs[0, target_class].item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        return cam, target_class, pred_prob


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay colored heatmap on grayscale image."""
    if image.ndim == 2:
        img3 = np.stack([image]*3, -1)
    elif image.shape[0] == 1:
        img3 = np.stack([image[0]]*3, -1)
    else:
        img3 = image
    hm_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1 - alpha) * img3 + alpha * hm_color, 0, 1)


# ============================================================================
# Cell 23: Grad-CAM Visualization (4-column layout)
# ============================================================================

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

grad_cam = GradCAM(model, target_layer=model.attention.spatial)

n_per_class = 5
benign_idx    = [i for i, l in enumerate(valid_ds.labels) if l == 0]
malignant_idx = [i for i, l in enumerate(valid_ds.labels) if l == 1]
np.random.seed(42)
sel = np.concatenate([
    np.random.choice(benign_idx,    min(n_per_class, len(benign_idx)),    replace=False),
    np.random.choice(malignant_idx, min(n_per_class, len(malignant_idx)), replace=False)
])

fig = plt.figure(figsize=(20, 4 * len(sel)))
gs  = gridspec.GridSpec(len(sel), 4, wspace=0.05, hspace=0.3)

for row, idx in enumerate(sel):
    img_t, msk_t, true_lbl = valid_ds[idx]
    inp = img_t.unsqueeze(0).to(device).requires_grad_(True)
    msk = msk_t.unsqueeze(0).to(device)

    heatmap, pred_cls, pred_prob = grad_cam.generate(inp, msk)
    orig = img_t[0].numpy()
    overlay = overlay_heatmap(orig, heatmap)
    mask_np = msk_t[0].numpy()

    ok = pred_cls == true_lbl
    color = 'green' if ok else 'red'
    tag   = 'Correct' if ok else 'Wrong'

    for ci, (im, cmap_, ttl) in enumerate([
        (orig,    'gray', f'Original\nTrue: {CLASS_NAMES[true_lbl]}'),
        (mask_np, 'gray', 'Breast Mask'),
        (overlay, None,   f'Grad-CAM Overlay\nPred: {CLASS_NAMES[pred_cls]} ({pred_prob:.1%})'),
        (heatmap, 'jet',  f'Heatmap\n{tag}'),
    ]):
        ax = fig.add_subplot(gs[row, ci])
        if cmap_: ax.imshow(im, cmap=cmap_, vmin=0, vmax=1)
        else:     ax.imshow(im)
        ax.set_title(ttl, fontsize=9); ax.axis('off')
        if ci in [0, 2, 3]:
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(3); sp.set_visible(True)

plt.suptitle('Mask-Guided Attention Visualization (Grad-CAM)\n'
             'Green=Correct, Red=Wrong', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'gradcam_v2.png'), dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# Cell 24: Attention With Mask vs Without Mask Comparison
# ============================================================================

print("=" * 70)
print("Comparison: Attention WITH mask guidance vs WITHOUT mask guidance")
print("=" * 70)

n_cmp = 4
cmp_idx = np.random.choice(sel, min(n_cmp, len(sel)), replace=False)

fig, axes = plt.subplots(n_cmp, 4, figsize=(18, 4.5 * n_cmp))
if n_cmp == 1: axes = axes.reshape(1, -1)

for row, idx in enumerate(cmp_idx):
    img_t, msk_t, true_lbl = valid_ds[idx]
    orig = img_t[0].numpy()
    inp  = img_t.unsqueeze(0).to(device).requires_grad_(True)
    msk  = msk_t.unsqueeze(0).to(device)

    # With mask
    hm_with, _, p_with = grad_cam.generate(inp.clone().requires_grad_(True), msk)
    ov_with = overlay_heatmap(orig, hm_with)

    # Without mask (pass None)
    hm_without, _, p_without = grad_cam.generate(inp.clone().requires_grad_(True), None)
    ov_without = overlay_heatmap(orig, hm_without)

    diff = hm_with - hm_without

    axes[row, 0].imshow(orig, cmap='gray')
    axes[row, 0].set_title(f'Original\nTrue: {CLASS_NAMES[true_lbl]}', fontsize=9); axes[row, 0].axis('off')
    axes[row, 1].imshow(ov_without)
    axes[row, 1].set_title(f'No Mask Guidance\n(p={p_without:.1%})', fontsize=9); axes[row, 1].axis('off')
    axes[row, 2].imshow(ov_with)
    axes[row, 2].set_title(f'Mask-Guided\n(p={p_with:.1%})', fontsize=9); axes[row, 2].axis('off')
    im = axes[row, 3].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[row, 3].set_title('Diff (Red=mask enhanced)', fontsize=9); axes[row, 3].axis('off')
    plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

plt.suptitle('Mask-Guided vs Unguided Attention Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'mask_vs_no_mask_attention.png'), dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# Cell 25: Attention Statistics
# ============================================================================

n_analysis = min(50, len(valid_ds))
analysis_idx = np.random.choice(len(valid_ds), n_analysis, replace=False)

att_stats = {'correct': {'in_breast': [], 'max': [], 'spread': []},
             'wrong':   {'in_breast': [], 'max': [], 'spread': []}}

grad_cam_stat = GradCAM(model, target_layer=model.attention.spatial)

for idx in tqdm(analysis_idx, desc="Analyzing attention"):
    img_t, msk_t, true_lbl = valid_ds[idx]
    inp = img_t.unsqueeze(0).to(device).requires_grad_(True)
    msk = msk_t.unsqueeze(0).to(device)
    hm, pred, _ = grad_cam_stat.generate(inp, msk)
    mask_np = msk_t[0].numpy()
    mask_resized = cv2.resize(mask_np, (hm.shape[1], hm.shape[0]), interpolation=cv2.INTER_NEAREST)

    key = 'correct' if pred == true_lbl else 'wrong'
    in_breast = hm[mask_resized > 0].sum() / (hm.sum() + 1e-8)
    att_stats[key]['in_breast'].append(in_breast)
    att_stats[key]['max'].append(hm.max())
    att_stats[key]['spread'].append(hm.std())

for key in ['correct', 'wrong']:
    n = len(att_stats[key]['in_breast'])
    if n > 0:
        tag = 'Correct' if key == 'correct' else 'Wrong'
        print(f"\n{tag} predictions (n={n}):")
        print(f"  In-breast attention ratio: {np.mean(att_stats[key]['in_breast']):.3f} +/- {np.std(att_stats[key]['in_breast']):.3f}")
        print(f"  Max attention value:       {np.mean(att_stats[key]['max']):.3f} +/- {np.std(att_stats[key]['max']):.3f}")
        print(f"  Attention spread (std):    {np.mean(att_stats[key]['spread']):.3f} +/- {np.std(att_stats[key]['spread']):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (metric, title) in enumerate([('in_breast', 'In-Breast Attention Ratio'),
                                      ('max', 'Max Attention'), ('spread', 'Attention Spread')]):
    data, lbls = [], []
    for key, lb in [('correct', f'Correct\n(n={len(att_stats["correct"][metric])})'),
                    ('wrong',   f'Wrong\n(n={len(att_stats["wrong"][metric])})')]:
        if att_stats[key][metric]:
            data.append(att_stats[key][metric]); lbls.append(lb)
    if data:
        bp = axes[i].boxplot(data, labels=lbls, patch_artist=True)
        for p, c in zip(bp['boxes'], ['#4CAF50', '#F44336'][:len(data)]):
            p.set_facecolor(c); p.set_alpha(0.6)
    axes[i].set_title(title); axes[i].grid(True, alpha=0.3)

plt.suptitle('Attention Analysis: Correct vs Wrong Predictions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'attention_stats_v2.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\nInterpretation:")
print("  - In-breast ratio close to 1.0 = attention stays inside breast (good)")
print("  - If wrong predictions have low in-breast ratio = model looks at background (bad)")
print("  - High spread in wrong predictions = unfocused attention")

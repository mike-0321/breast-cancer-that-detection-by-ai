# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Install Dependencies ---
# !pip install -q monai scikit-learn opencv-python-headless

# --- Import Libraries ---
import os, glob, shutil, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sns

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from monai.transforms import Compose, EnsureType
from monai.networks.nets import DenseNet121
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# --- Path Configuration ---
base_dir        = '/content/drive/MyDrive/breast cancer'
train_dir       = os.path.join(base_dir, 'train')
valid_dir       = os.path.join(base_dir, 'valid')
test_dir        = os.path.join(base_dir, 'test')
processed_dir   = os.path.join(base_dir, 'processed_v2_1')
processed_train = os.path.join(processed_dir, 'train')
processed_valid = os.path.join(processed_dir, 'valid')
processed_test  = os.path.join(processed_dir, 'test')
model_path      = '/content/drive/MyDrive/best_model_v2_1.pth'
IMG_SIZE        = 224


# --- Image Loading ---

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


# --- White-BG Detection ---

def is_white_background(image, wr_thresh=0.3, br_thresh=0.5, mean_thresh=0.7):
    """Returns True if image has a white background."""
    wr = (image > 0.95).sum() / image.size
    br = (image > 0.80).sum() / image.size
    if wr > wr_thresh or br > br_thresh:
        return True
    if image.mean() > mean_thresh and image.std() < 0.2:
        return True
    return False


# --- Breast Mask (ONLY for bounding box localization) ---

def create_breast_mask(image):
    """
    Create binary breast mask using Otsu + largest connected component.
    Used ONLY to find bounding box — never applied to zero out pixels.
    """
    img_uint8 = (image * 255).astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num_labels <= 1:
        return (binary > 0).astype(np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


# --- Crop + Pad — NO internal pixel modification ---
# Update from v2: Removed image * mask — original pixels fully preserved.
# Update from v2: No pectoral removal — all internal breast info kept.

def crop_and_pad(image, mask, target_size=224):
    """
    Use mask bounding box to crop the breast from the ORIGINAL image.
    Pad to square (preserve aspect ratio), then resize.
    Original pixels are NEVER zeroed out or modified.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        r0, c0, r1, c1 = 0, 0, image.shape[0], image.shape[1]
    else:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1

    # Crop ORIGINAL image (not masked)
    cropped_img  = image[r0:r1, c0:c1]
    cropped_mask = mask[r0:r1, c0:c1]

    # Pad to square
    ch, cw = cropped_img.shape
    ms = max(ch, cw)
    ph, pw = (ms - ch) // 2, (ms - cw) // 2

    padded_img  = np.zeros((ms, ms), dtype=np.float32)
    padded_mask = np.zeros((ms, ms), dtype=np.uint8)
    padded_img[ph:ph+ch, pw:pw+cw]  = cropped_img
    padded_mask[ph:ph+ch, pw:pw+cw] = cropped_mask

    proc_img  = cv2.resize(padded_img,  (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    proc_mask = cv2.resize(padded_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return proc_img, proc_mask


# --- Batch Preprocess + Save ---

def preprocess_and_save_all(src_dir, dst_dir, target_size=224):
    """Load → White-BG filter → Mask (bbox only) → Crop → Pad → Save .npz"""
    total = {'processed': 0, 'white_skipped': 0, 'failed': 0, 'exist': 0}
    for cls in ['0', '1']:
        src_cls = os.path.join(src_dir, cls)
        dst_cls = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)
        files = glob.glob(os.path.join(src_cls, '*'))
        print(f"\n  {os.path.basename(src_dir)}/class_{cls}: {len(files)} files")
        for f in tqdm(files, desc=f"Class {cls}"):
            fname = os.path.splitext(os.path.basename(f))[0]
            save_path = os.path.join(dst_cls, f"{fname}.npz")
            if os.path.exists(save_path):
                total['exist'] += 1; continue
            image = load_image_grayscale(f)
            if image is None:
                total['failed'] += 1; continue
            if is_white_background(image):
                total['white_skipped'] += 1; continue
            mask = create_breast_mask(image)
            proc_img, proc_mask = crop_and_pad(image, mask, target_size)
            np.savez_compressed(save_path, image=proc_img, mask=proc_mask)
            total['processed'] += 1
    print(f"\n  Done: {total}")
    return total

if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
    print("Cleared old data.\n")

print("=" * 70)
print("Step 1: Preprocess (white-BG filter + crop, NO internal masking)")
print("=" * 70)
for name, src, dst in [('train', train_dir, processed_train),
                        ('valid', valid_dir, processed_valid),
                        ('test',  test_dir,  processed_test)]:
    print(f"\n--- {name.upper()} ---")
    preprocess_and_save_all(src, dst, IMG_SIZE)


# --- Visualize Preprocessing ---

def visualize_pipeline(src_dir, dst_dir, cls='1', n=4):
    pairs = []
    for sf in glob.glob(os.path.join(src_dir, cls, '*')):
        fname = os.path.splitext(os.path.basename(sf))[0]
        npz = os.path.join(dst_dir, cls, f"{fname}.npz")
        if os.path.exists(npz):
            pairs.append((sf, npz))
        if len(pairs) >= n: break
    fig, axes = plt.subplots(len(pairs), 3, figsize=(12, 4*len(pairs)))
    if len(pairs) == 1: axes = axes.reshape(1, -1)
    fig.suptitle('Preprocessing: Original → Mask (bbox) → Cropped+Padded (pixels preserved)',
                 fontsize=12, fontweight='bold')
    for i, (sf, npz) in enumerate(pairs):
        orig = load_image_grayscale(sf)
        data = np.load(npz)
        axes[i,0].imshow(orig, cmap='gray'); axes[i,0].set_title(f'Original {orig.shape}'); axes[i,0].axis('off')
        axes[i,1].imshow(create_breast_mask(orig), cmap='gray'); axes[i,1].set_title('Mask (bbox locator)'); axes[i,1].axis('off')
        axes[i,2].imshow(data['image'], cmap='gray'); axes[i,2].set_title(f'Result {data["image"].shape}'); axes[i,2].axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(base_dir, 'preprocess_v2_1.png'), dpi=150, bbox_inches='tight'); plt.show()

visualize_pipeline(train_dir, processed_train, '1', 4)


# --- Build File Lists + Dataset ---

def build_file_list(data_dir):
    files, labels = [], []
    for cls in ['0', '1']:
        fs = glob.glob(os.path.join(data_dir, cls, '*.npz'))
        files.extend(fs); labels.extend([int(cls)] * len(fs))
    return files, labels

train_files, train_labels = build_file_list(processed_train)
valid_files, valid_labels = build_file_list(processed_valid)
test_files,  test_labels  = build_file_list(processed_test)
print(f'Train: {len(train_files)} (c0={train_labels.count(0)}, c1={train_labels.count(1)})')
print(f'Valid: {len(valid_files)} | Test: {len(test_files)}')


class MammogramDataset(Dataset):
    """Loads .npz (image + mask). Returns (1,H,W) image + (1,H,W) mask + label."""
    def __init__(self, files, labels, augment=False):
        self.files = files; self.labels = labels; self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = data['image'].astype(np.float32)
        mask  = data['mask'].astype(np.float32)
        label = self.labels[idx]
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, 0).copy(); mask = np.flip(mask, 0).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, 1).copy(); mask = np.flip(mask, 1).copy()
            if np.random.rand() > 0.5:
                k = np.random.choice([1,2,3])
                image = np.rot90(image, k).copy(); mask = np.rot90(mask, k).copy()
            if np.random.rand() > 0.7:
                image = np.clip(image + np.random.uniform(-0.05, 0.05), 0, 1)
            if np.random.rand() > 0.8:
                image = np.clip(image + np.random.randn(*image.shape)*0.03, 0, 1).astype(np.float32)
        return (torch.from_numpy(image[None,...]),
                torch.from_numpy(mask[None,...]),
                label)


# --- DataLoader ---
BATCH_SIZE = 16; NUM_WORKERS = 2

train_ds = MammogramDataset(train_files, train_labels, augment=True)
valid_ds = MammogramDataset(valid_files, valid_labels, augment=False)
test_ds  = MammogramDataset(test_files,  test_labels,  augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

img, msk, lbl = train_ds[0]
print(f'Sample: image={img.shape}, mask={msk.shape}, label={lbl}')


# --- Attention Modules ---
# Update from v2: Soft-guided attention replaces hard mask gating.
# Added: Anomaly detector: finds hotspot regions from attention map.
# Added: Local feature extractor: crops and pools anomaly region features.

class SoftGuidedSpatialAttention(nn.Module):
    """
    Spatial attention with soft mask guidance.
    Learnable alpha controls mask influence:
      alpha ~ 0: pure self-attention (ignores mask)
      alpha ~ 1: strongly mask-guided
    Breast boundary info is NEVER destroyed.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, mask=None):
        att = self.spatial_conv(x)  # (B, 1, H', W')
        if mask is not None:
            mask_down = F.interpolate(mask, size=att.shape[2:],
                                      mode='bilinear', align_corners=False)
            alpha = torch.sigmoid(self.alpha_raw)
            soft_gate = alpha * mask_down + (1.0 - alpha)
            att = att * soft_gate
        return x * att, att  # Return attention map for anomaly detection


class AnomalyRegionDetector(nn.Module):
    """
    Detects anomaly hotspot regions from the spatial attention map.

    How it works:
      1. Take the attention map (B, 1, H', W')
      2. Find top-K peak locations (highest attention values)
      3. Generate bounding boxes around each peak
      4. Use ROI-align-like cropping to extract local features from those boxes

    This forces the model to explicitly "look at" the most suspicious regions,
    rather than just averaging everything through global average pooling.
    """
    def __init__(self, feature_channels, roi_size=3, top_k=3):
        super().__init__()
        self.roi_size = roi_size
        self.top_k = top_k
        # Local feature compressor: pool each ROI into a fixed-size vector
        self.local_pool = nn.AdaptiveAvgPool2d(1)
        self.local_fc = nn.Sequential(
            nn.Linear(feature_channels, feature_channels // 2),
            nn.ReLU(inplace=False),
            nn.Linear(feature_channels // 2, feature_channels // 4),
        )

    def _find_topk_peaks(self, att_map):
        """
        Find top-K peak locations in attention map.
        Args: att_map (B, 1, H, W)
        Returns: list of (batch_index, row, col) tuples
        """
        B, _, H, W = att_map.shape
        flat = att_map.view(B, -1)
        _, topk_idx = flat.topk(self.top_k, dim=1)

        peaks = []
        for b in range(B):
            for k in range(self.top_k):
                idx = topk_idx[b, k].item()
                r, c = idx // W, idx % W
                peaks.append((b, r, c))
        return peaks

    def _crop_roi(self, features, b, r, c):
        """Crop a local region around (r, c) from feature map."""
        _, _, H, W = features.shape
        half = self.roi_size // 2
        r0 = max(0, r - half)
        r1 = min(H, r + half + 1)
        c0 = max(0, c - half)
        c1 = min(W, c + half + 1)
        roi = features[b:b+1, :, r0:r1, c0:c1]
        return roi

    def forward(self, features, att_map):
        """
        Args:
            features: (B, C, H', W') feature maps
            att_map:  (B, 1, H', W') spatial attention map
        Returns:
            local_features: (B, C//4) aggregated local anomaly features
            peaks_info:     list of peak locations for visualization
        """
        B, C, H, W = features.shape
        peaks = self._find_topk_peaks(att_map.detach())

        # Collect local features from each peak
        local_feats = []
        for b in range(B):
            batch_peaks = [(r, c) for (bi, r, c) in peaks if bi == b]
            roi_feats = []
            for (r, c) in batch_peaks:
                roi = self._crop_roi(features, b, r, c)
                pooled = self.local_pool(roi).flatten()  # (C,)
                roi_feats.append(pooled)

            if roi_feats:
                # Average features from all K peaks
                stacked = torch.stack(roi_feats, dim=0)  # (K, C)
                aggregated = stacked.mean(dim=0)          # (C,)
            else:
                aggregated = features.new_zeros(C)
            local_feats.append(aggregated)

        local_feats = torch.stack(local_feats, dim=0)  # (B, C)
        local_feats = self.local_fc(local_feats)         # (B, C//4)

        return local_feats, peaks


# --- Model Definition — Global + Local Dual-Branch ---
# Update from v2: Hard mask attention → Soft-guided attention
# Added: Anomaly detector extracts local features from hotspot regions
# Added: Global + Local fusion: classifier uses both branches
# Update from v2: in_channels=1 (no 3-ch expansion)

class DenseNetAnomalyAware(nn.Module):
    """
    DenseNet121 + Soft-Guided Attention + Anomaly-Aware Local Branch.

    Architecture:
      Input (1, 224, 224) + Mask (1, 224, 224)
        → DenseNet121 backbone → features (1024, 7, 7)
        → Soft-guided spatial attention → att_map (1, 7, 7)
        → Global branch: GAP on full attended features → (1024,)
        → Local branch: detect top-3 hotspots → crop ROI → pool → (256,)
        → Fuse: concat(global, local) → (1280,) → FC → 2 classes

    The local branch forces the model to explicitly focus on the most
    suspicious regions, not just rely on global average pooling which
    can dilute small lesion signals.
    """
    def __init__(self, num_classes=2, dropout_rate=0.4, top_k=3):
        super().__init__()
        # Backbone
        backbone = DenseNet121(spatial_dims=2, in_channels=1, out_channels=1024)
        self.features = backbone.features

        # Soft-guided spatial attention
        self.spatial_attention = SoftGuidedSpatialAttention(in_channels=1024)

        # Anomaly region detector + local feature extractor
        self.anomaly_detector = AnomalyRegionDetector(
            feature_channels=1024, roi_size=3, top_k=top_k)

        # Global branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )  # → (B, 1024)

        # Fusion classifier: global (1024) + local (256) = 1280
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024 + 256, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, mask=None):
        feat = self.features(x)                          # (B, 1024, 7, 7)
        attended_feat, att_map = self.spatial_attention(feat, mask)

        # Global branch
        global_feat = self.global_pool(attended_feat)     # (B, 1024)

        # Local branch: detect anomaly hotspots and extract features
        local_feat, peaks = self.anomaly_detector(attended_feat, att_map)  # (B, 256)

        # Fuse global + local
        fused = torch.cat([global_feat, local_feat], dim=1)  # (B, 1280)
        logits = self.classifier(fused)

        return logits, att_map, peaks  # Return extra info for visualization


model = DenseNetAnomalyAware(num_classes=2, dropout_rate=0.4, top_k=3).to(device)

def disable_inplace(m):
    for c in m.children():
        if isinstance(c, (nn.ReLU, nn.SiLU)):
            c.inplace = False
        else:
            disable_inplace(c)
disable_inplace(model)
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")


# --- Training Configuration ---
MAX_EPOCHS = 30; LR = 1e-4; WEIGHT_DECAY = 1e-4; PATIENCE = 10

n0, n1 = train_labels.count(0), train_labels.count(1)
w0, w1 = (n0+n1)/(2*n0), (n0+n1)/(2*n1)
class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
print(f'Class weights -> 0: {w0:.4f}, 1: {w1:.4f}')

loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)


# --- Training Loop ---
# Update from v2: model returns (logits, att_map, peaks) — only use logits for loss
# Added: Prints learned alpha value each epoch to monitor mask influence

train_losses, val_accs, val_losses, lr_history = [], [], [], []
best_val_acc = 0.0; patience_count = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss, steps = 0.0, 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{MAX_EPOCHS}]")
    for imgs, masks, labels in loop:
        imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(imgs, masks)  # Ignore att_map and peaks during training
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item(); steps += 1
        loop.set_postfix(avg_loss=epoch_loss/steps)

    avg_train = epoch_loss / steps
    train_losses.append(avg_train)
    lr_history.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

    model.eval()
    correct, n_val, vl_sum = 0, 0, 0.0
    with torch.no_grad():
        for imgs, masks, labels in valid_loader:
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            logits, _, _ = model(imgs, masks)
            vl_sum  += loss_fn(logits, labels).item()
            correct += (logits.argmax(1) == labels).sum().item()
            n_val   += len(labels)

    val_acc = correct / n_val
    avg_val = vl_sum / len(valid_loader)
    val_accs.append(val_acc); val_losses.append(avg_val)

    alpha = torch.sigmoid(model.spatial_attention.alpha_raw).item()
    print(f"Epoch [{epoch+1:02d}] Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
          f"Acc: {val_acc:.4f} | LR: {lr_history[-1]:.2e} | alpha: {alpha:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f"  Best saved ({best_val_acc:.4f})")
        patience_count = 0
    else:
        patience_count += 1
    if patience_count >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}"); break

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
print(f"Final alpha: {torch.sigmoid(model.spatial_attention.alpha_raw).item():.3f}")


# --- Training Curves ---
x = range(1, len(train_losses)+1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(x, train_losses, 'o-', label='Train'); axes[0].plot(x, val_losses, 's-', label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
axes[1].plot(x, val_accs, 'o-'); axes[1].axhline(best_val_acc, ls='--', label=f'Best={best_val_acc:.4f}')
axes[1].set_title('Val Accuracy'); axes[1].legend(); axes[1].grid(True)
axes[2].plot(x, lr_history, 'o-'); axes[2].set_title('LR'); axes[2].grid(True)
plt.tight_layout(); plt.savefig(os.path.join(base_dir, 'curve_v2_1.png'), dpi=150); plt.show()


# --- Evaluation ---
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
disable_inplace(model)

all_preds, all_labels_e, all_probs = [], [], []
with torch.no_grad():
    for imgs, masks, labels in valid_loader:
        logits, _, _ = model(imgs.to(device), masks.to(device))
        all_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels_e.extend(labels.numpy())
all_preds = np.array(all_preds); all_labels_e = np.array(all_labels_e); all_probs = np.array(all_probs)


# --- Confusion Matrix + Report ---
CN = ['Benign', 'Malignant']
cm = confusion_matrix(all_labels_e, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CN, yticklabels=CN,
            linewidths=0.5, linecolor='black')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix (DenseNet121 + Anomaly-Aware Attention)')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'cm_v2_1.png'), dpi=150, bbox_inches='tight'); plt.show()

acc=accuracy_score(all_labels_e,all_preds); prec=precision_score(all_labels_e,all_preds,average='binary')
rec=recall_score(all_labels_e,all_preds,average='binary'); f1=f1_score(all_labels_e,all_preds,average='binary')
try: auc=roc_auc_score(all_labels_e,all_probs)
except: auc=None
print(f"\nAccuracy:{acc:.4f} | Precision:{prec:.4f} | Recall:{rec:.4f} | F1:{f1:.4f}")
if auc: print(f"AUC:{auc:.4f}")
print("\n"+classification_report(all_labels_e,all_preds,target_names=CN,digits=4))


# --- Grad-CAM + Anomaly Box Visualization ---
# Added: Draws red bounding boxes around detected anomaly hotspots
# Added: Shows zoomed-in crops of each anomaly region
# Update from v2: 5-column layout: Original | Mask | Heatmap+Boxes | Overlay | Local Crops

class GradCAMAnomaly:
    """Grad-CAM that also returns anomaly peak locations for box drawing."""
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None; self.gradients = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, img_t, mask_t, target_class=None):
        self.model.eval()
        logits, att_map, peaks = self.model(img_t, mask_t)
        probs = torch.softmax(logits, 1)
        if target_class is None:
            target_class = logits.argmax(1).item()
        prob = probs[0, target_class].item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        w = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((w * self.activations).sum(1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        h_in, w_in = img_t.shape[2], img_t.shape[3]
        cam_resized = cv2.resize(cam, (w_in, h_in))

        # Convert peaks from feature-map coords to image coords
        _, _, fh, fw = att_map.shape
        scale_h, scale_w = h_in / fh, w_in / fw
        roi_size = self.model.anomaly_detector.roi_size

        boxes = []
        for (b, r, c) in peaks:
            if b == 0:  # Only first image in batch
                half = roi_size // 2
                y0 = int(max(0, (r - half)) * scale_h)
                y1 = int(min(fh, (r + half + 1)) * scale_h)
                x0 = int(max(0, (c - half)) * scale_w)
                x1 = int(min(fw, (c + half + 1)) * scale_w)
                boxes.append((x0, y0, x1, y1))

        return cam_resized, target_class, prob, boxes


def overlay_heatmap(image, heatmap, alpha=0.4):
    if image.ndim == 2: img3 = np.stack([image]*3, -1)
    elif image.shape[0] == 1: img3 = np.stack([image[0]]*3, -1)
    else: img3 = image
    hm = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1-alpha)*img3 + alpha*hm, 0, 1)


# --- Full Visualization with Anomaly Boxes ---

grad_cam = GradCAMAnomaly(model, target_layer=model.spatial_attention.spatial_conv)

n_per = 5
b_idx = [i for i, l in enumerate(valid_ds.labels) if l == 0]
m_idx = [i for i, l in enumerate(valid_ds.labels) if l == 1]
np.random.seed(42)
sel = np.concatenate([
    np.random.choice(b_idx, min(n_per, len(b_idx)), replace=False),
    np.random.choice(m_idx, min(n_per, len(m_idx)), replace=False)
])

fig = plt.figure(figsize=(24, 4.5*len(sel)))
gs = gridspec.GridSpec(len(sel), 5, wspace=0.08, hspace=0.35)

for row, idx in enumerate(sel):
    img_t, msk_t, true_lbl = valid_ds[idx]
    inp = img_t.unsqueeze(0).to(device).requires_grad_(True)
    msk = msk_t.unsqueeze(0).to(device)

    hm, pred, prob, boxes = grad_cam.generate(inp, msk)
    orig = img_t[0].numpy()
    ov = overlay_heatmap(orig, hm)
    mask_np = msk_t[0].numpy()

    ok = pred == true_lbl
    col = 'green' if ok else 'red'
    tag = 'Correct' if ok else 'Wrong'

    # Col 0: Original
    ax = fig.add_subplot(gs[row, 0])
    ax.imshow(orig, cmap='gray'); ax.set_title(f'Original\nTrue: {CN[true_lbl]}', fontsize=9); ax.axis('off')

    # Col 1: Mask
    ax = fig.add_subplot(gs[row, 1])
    ax.imshow(mask_np, cmap='gray'); ax.set_title('Breast Mask\n(bbox locator)', fontsize=9); ax.axis('off')

    # Col 2: Overlay with anomaly boxes
    ax = fig.add_subplot(gs[row, 2])
    ax.imshow(ov)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        rect = patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                  linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x0, y0-3, f'#{i+1}', color='red', fontsize=8, fontweight='bold')
    ax.set_title(f'Grad-CAM + Anomaly Boxes\nPred: {CN[pred]} ({prob:.1%})', fontsize=9)
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(3); sp.set_visible(True)

    # Col 3: Heatmap with boxes
    ax = fig.add_subplot(gs[row, 3])
    ax.imshow(hm, cmap='jet', vmin=0, vmax=1)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        rect = patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                  linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(f'Heatmap + Boxes\n{tag}', fontsize=9); ax.axis('off')

    # Col 4: Zoomed anomaly crops
    ax = fig.add_subplot(gs[row, 4])
    if boxes:
        n_boxes = len(boxes)
        crop_h = max(1, 224 // n_boxes)
        combined = np.zeros((crop_h * n_boxes, crop_h, 3), dtype=np.float32)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            crop = orig[max(0,y0):min(224,y1), max(0,x0):min(224,x1)]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (crop_h, crop_h))
                crop_3ch = np.stack([crop_resized]*3, -1)
                combined[i*crop_h:(i+1)*crop_h, :, :] = crop_3ch
        ax.imshow(combined)
        ax.set_title(f'Anomaly Crops\n({n_boxes} regions)', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No peaks', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Anomaly Crops', fontsize=9)
    ax.axis('off')

plt.suptitle('DenseNet121 + Anomaly-Aware Soft Attention (Global + Local)\n'
             'Red boxes = detected anomaly hotspots | Green border = Correct, Red border = Wrong',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'gradcam_anomaly_v2_1.png'), dpi=150, bbox_inches='tight')
plt.show()


# --- Attention Statistics ---

n_ana = min(50, len(valid_ds))
ana_idx = np.random.choice(len(valid_ds), n_ana, replace=False)
att_s = {'correct': {'in_breast': [], 'max': [], 'std': [], 'n_peaks_in_breast': []},
         'wrong':   {'in_breast': [], 'max': [], 'std': [], 'n_peaks_in_breast': []}}

gc_stat = GradCAMAnomaly(model, target_layer=model.spatial_attention.spatial_conv)

for idx in tqdm(ana_idx, desc="Analyzing"):
    img_t, msk_t, tl = valid_ds[idx]
    hm, pred, _, boxes = gc_stat.generate(
        img_t.unsqueeze(0).to(device).requires_grad_(True),
        msk_t.unsqueeze(0).to(device))
    mk = msk_t[0].numpy()
    key = 'correct' if pred == tl else 'wrong'
    att_s[key]['in_breast'].append(hm[mk > 0].sum() / (hm.sum() + 1e-8))
    att_s[key]['max'].append(hm.max())
    att_s[key]['std'].append(hm.std())
    # Count how many anomaly boxes fall inside the breast mask
    in_breast = sum(1 for (x0,y0,x1,y1) in boxes
                    if mk[min(y0+((y1-y0)//2), 223), min(x0+((x1-x0)//2), 223)] > 0)
    att_s[key]['n_peaks_in_breast'].append(in_breast / max(len(boxes), 1))

for key in ['correct', 'wrong']:
    n = len(att_s[key]['in_breast'])
    if n > 0:
        tag = 'CORRECT' if key == 'correct' else 'WRONG'
        print(f"\n{tag} (n={n}):")
        print(f"  In-breast att ratio:    {np.mean(att_s[key]['in_breast']):.3f} +/- {np.std(att_s[key]['in_breast']):.3f}")
        print(f"  Max attention:          {np.mean(att_s[key]['max']):.3f} +/- {np.std(att_s[key]['max']):.3f}")
        print(f"  Spread (std):           {np.mean(att_s[key]['std']):.3f} +/- {np.std(att_s[key]['std']):.3f}")
        print(f"  Peaks-in-breast ratio:  {np.mean(att_s[key]['n_peaks_in_breast']):.3f} +/- {np.std(att_s[key]['n_peaks_in_breast']):.3f}")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, (m, t) in enumerate([('in_breast','In-Breast Ratio'),('max','Max Attention'),
                             ('std','Spread'),('n_peaks_in_breast','Peaks in Breast')]):
    d, lb = [], []
    for k, l in [('correct',f'Correct (n={len(att_s["correct"][m])})'),
                 ('wrong',  f'Wrong (n={len(att_s["wrong"][m])})')]:
        if att_s[k][m]: d.append(att_s[k][m]); lb.append(l)
    if d:
        bp = axes[i].boxplot(d, labels=lb, patch_artist=True)
        for p, c in zip(bp['boxes'], ['#4CAF50','#F44336'][:len(d)]):
            p.set_facecolor(c); p.set_alpha(0.6)
    axes[i].set_title(t); axes[i].grid(True, alpha=0.3)

plt.suptitle('Attention Analysis: Correct vs Wrong', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'att_stats_v2_1.png'), dpi=150, bbox_inches='tight'); plt.show()

alpha_val = torch.sigmoid(model.spatial_attention.alpha_raw).item()
print(f"\nLearned alpha: {alpha_val:.3f}")
print("  alpha ~0 = model ignores mask | alpha ~1 = heavily mask-guided")
print("\nNew metric - Peaks-in-breast ratio:")
print("  Close to 1.0 = anomaly hotspots are inside breast (good)")
print("  Low value = model detecting 'anomalies' in background (bad, misleading)")

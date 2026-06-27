import os
import numpy as np
import torch
from collections import Counter

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Resize,
    ScaleIntensity,
    RandRotate,
    RandFlip,
    RandZoom,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandHistogramShift,
    RandBiasField,
    RandAdjustContrast,
)
from monai.data import ImageDataset, DataLoader
from monai.networks.nets import DenseNet121
from monai.losses import FocalLoss


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
VAL_DIR    = os.path.join(DATA_DIR, "val")
MODEL_PATH = "models/fetal_ultrasound_model.pth"

IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 8
NUM_EPOCHS  = 30
LR          = 1e-4
PATIENCE    = 10
N_FOLDS     = 5

CLASSES = [
    "trans_thalamic",
    "trans_ventricular",
    "trans_cerebellum",
    "diverse",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create models directory once at the start — avoids repeated makedirs errors
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build file lists
# ─────────────────────────────────────────────────────────────────────────────
def build_file_list(root_dir):
    files, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.exists(cls_path):
            print(f"  WARNING: class folder not found -> {cls_path}")
            continue
        for img in sorted(os.listdir(cls_path)):
            if img.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append(os.path.join(cls_path, img))
                labels.append(idx)
    return files, labels


train_files, train_labels = build_file_list(TRAIN_DIR)
val_files,   val_labels   = build_file_list(VAL_DIR)

all_files  = np.array(train_files + val_files)
all_labels = np.array(train_labels + val_labels)

print(f"Total images for k-fold: {len(all_files)}")
print(f"Class distribution: {dict(Counter(all_labels.tolist()))}")


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
train_transforms = Compose([
    EnsureChannelFirst(channel_dim=-1),
    Resize(IMAGE_SIZE),
    ScaleIntensity(),
    RandRotate(range_x=0.15, prob=0.5),
    RandFlip(spatial_axis=0, prob=0.4),
    RandFlip(spatial_axis=1, prob=0.4),
    RandZoom(min_zoom=0.85, max_zoom=1.15, prob=0.3),
    RandGaussianNoise(prob=0.2, std=0.02),
    RandGaussianSmooth(prob=0.2),
    RandHistogramShift(prob=0.2),
    RandBiasField(prob=0.2),
    RandAdjustContrast(prob=0.3),
])

val_transforms = Compose([
    EnsureChannelFirst(channel_dim=-1),
    Resize(IMAGE_SIZE),
    ScaleIntensity(),
])


# ─────────────────────────────────────────────────────────────────────────────
# Helper — build fresh model each fold
# ─────────────────────────────────────────────────────────────────────────────
def build_model():
    return DenseNet121(
        spatial_dims=2,
        in_channels=3,
        out_channels=4,
    ).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold Cross Validation
# ─────────────────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_accuracies       = []
fold_val_losses       = []
all_fold_preds        = []
all_fold_labels       = []
best_overall_val_loss = float("inf")

print(f"\n{'='*60}")
print(f"  Starting {N_FOLDS}-Fold Stratified Cross Validation")
print(f"{'='*60}\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels)):

    print(f"\n{'─'*60}")
    print(f"  FOLD {fold+1} / {N_FOLDS}")
    print(f"{'─'*60}")
    print(f"  Train: {len(train_idx)} images | Val: {len(val_idx)} images")

    # ── Split ─────────────────────────────────────────────────────────────────
    fold_train_files  = all_files[train_idx].tolist()
    fold_train_labels = all_labels[train_idx].tolist()
    fold_val_files    = all_files[val_idx].tolist()
    fold_val_labels   = all_labels[val_idx].tolist()

    # ── Class weights ─────────────────────────────────────────────────────────
    fold_class_counts = Counter(fold_train_labels)
    fold_total        = sum(fold_class_counts.values())
    fold_weights      = torch.tensor(
        [fold_total / fold_class_counts[i] for i in range(len(CLASSES))],
        dtype=torch.float
    ).to(device)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    fold_train_ds = ImageDataset(
        image_files=fold_train_files,
        labels=fold_train_labels,
        transform=train_transforms
    )
    fold_val_ds = ImageDataset(
        image_files=fold_val_files,
        labels=fold_val_labels,
        transform=val_transforms
    )
    fold_train_loader = DataLoader(fold_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    fold_val_loader   = DataLoader(fold_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    model     = build_model()
    criterion = FocalLoss(gamma=2.0, weight=fold_weights, reduction="mean", to_onehot_y=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_fold_val_loss = float("inf")
    epochs_no_improve  = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):

        # Train
        model.train()
        epoch_train_loss = 0.0
        for images, labels_batch in fold_train_loader:
            images       = images.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(fold_train_loader)

        # Validate
        model.eval()
        epoch_val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for images, labels_batch in fold_val_loader:
                images       = images.to(device)
                labels_batch = labels_batch.to(device)
                outputs      = model(images)
                loss         = criterion(outputs, labels_batch)
                epoch_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels_batch.cpu().numpy())

        epoch_val_loss /= len(fold_val_loader)
        val_acc    = accuracy_score(val_true, val_preds)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"  Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  "
            f"Train Loss: {epoch_train_loss:.4f}  "
            f"Val Loss: {epoch_val_loss:.4f}  "
            f"Val Acc: {val_acc*100:.2f}%  "
            f"LR: {current_lr:.2e}"
        )

        scheduler.step(epoch_val_loss)

        # Checkpoint
        if epoch_val_loss < best_fold_val_loss:
            best_fold_val_loss = epoch_val_loss
            epochs_no_improve  = 0
            torch.save(model.state_dict(), f"models/fold_{fold+1}_best.pth")
            print(f"  Fold {fold+1} best model saved (val loss {best_fold_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  Early stopping triggered after {PATIENCE} epochs no improvement.")
                break

    # ── Evaluate this fold ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(f"models/fold_{fold+1}_best.pth", map_location=device))
    model.eval()

    fold_preds, fold_true = [], []
    with torch.no_grad():
        for images, labels_batch in fold_val_loader:
            images  = images.to(device)
            outputs = model(images)
            preds   = torch.argmax(outputs, dim=1)
            fold_preds.extend(preds.cpu().numpy())
            fold_true.extend(labels_batch.numpy())

    fold_acc = accuracy_score(fold_true, fold_preds)
    fold_accuracies.append(fold_acc)
    fold_val_losses.append(best_fold_val_loss)
    all_fold_preds.extend(fold_preds)
    all_fold_labels.extend(fold_true)

    print(f"\n  Fold {fold+1} Final Accuracy: {fold_acc*100:.2f}%")

    # Save best overall model
    if best_fold_val_loss < best_overall_val_loss:
        best_overall_val_loss = best_fold_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  New best overall model saved from Fold {fold+1}")


# ─────────────────────────────────────────────────────────────────────────────
# K-Fold Summary
# ─────────────────────────────────────────────────────────────────────────────
mean_acc = np.mean(fold_accuracies)
std_acc  = np.std(fold_accuracies)

print(f"\n{'='*60}")
print(f"  {N_FOLDS}-FOLD CROSS VALIDATION RESULTS")
print(f"{'='*60}")
for i, acc in enumerate(fold_accuracies):
    print(f"  Fold {i+1}: {acc*100:.2f}%")
print(f"\n  Mean Accuracy : {mean_acc*100:.2f}%")
print(f"  Std Deviation : +/-{std_acc*100:.2f}%")
print(f"\n  Report in paper as: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Final Combined Evaluation
# ─────────────────────────────────────────────────────────────────────────────
final_acc = accuracy_score(all_fold_labels, all_fold_preds)
print(f"Overall Accuracy (all folds combined): {final_acc*100:.2f}%")

print("\nClassification Report:")
print(classification_report(all_fold_labels, all_fold_preds, target_names=CLASSES))

cm = confusion_matrix(all_fold_labels, all_fold_preds)
print("Confusion Matrix:")
print(cm)

print("\nPer-class Accuracy:")
for i, cls in enumerate(CLASSES):
    cls_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"  {cls:25s}: {cls_acc*100:.2f}%")

print(f"\nBest model saved to: {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Per-fold accuracy
fold_names  = [f"Fold {i+1}" for i in range(N_FOLDS)]
bar_colors  = ["#22c55e" if acc >= mean_acc else "#3b82f6" for acc in fold_accuracies]
axes[0].bar(fold_names, [a * 100 for a in fold_accuracies], color=bar_colors, edgecolor="white", linewidth=1.5)
axes[0].axhline(y=mean_acc * 100, color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: {mean_acc*100:.2f}%")
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title(f"5-Fold Cross Validation\nMean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
axes[0].legend()
axes[0].set_ylim(50, 100)
axes[0].grid(True, alpha=0.3)

# 2. Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")
axes[1].set_title(f"Confusion Matrix (All Folds)\nOverall Acc: {final_acc*100:.1f}%")

# 3. Per-class accuracy
per_class_accs = []
for i in range(len(CLASSES)):
    cls_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    per_class_accs.append(cls_acc * 100)

class_colors = ["#22c55e" if a >= 80 else "#f59e0b" if a >= 60 else "#ef4444" for a in per_class_accs]
axes[2].barh(CLASSES, per_class_accs, color=class_colors, edgecolor="white", linewidth=1.5)
axes[2].axvline(x=80, color="#64748b", linestyle="--", linewidth=1.5, label="80% threshold")
axes[2].set_xlabel("Accuracy (%)")
axes[2].set_title("Per-Class Accuracy")
axes[2].set_xlim(0, 100)
axes[2].legend()
axes[2].grid(True, alpha=0.3)
for i, acc in enumerate(per_class_accs):
    axes[2].text(acc + 1, i, f"{acc:.1f}%", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig("models/kfold_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to models/kfold_results.png")
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.brats_dataset import BraTSSliceDataset
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from peft import PeftModel
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import imageio
import json
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -------------------------------
# Model Definition (MUST match training)
# -------------------------------
class SimpleSegHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=4):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        return x

# -------------------------------
# Load model
# -------------------------------
print("\nLoading model...")
base_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
base_model.from_pretrained()
vision_model = base_model.vision_model
vision_model = PeftModel.from_pretrained(vision_model, "./checkpoints/lora_medclip")
vision_model.to(device).eval()

seg_head = SimpleSegHead(in_channels=512, num_classes=4)
seg_head.load_state_dict(torch.load("./checkpoints/seg_head.pth", map_location=device))
seg_head.to(device).eval()
print("✓ Model loaded\n")

# -------------------------------
# Dataset
# -------------------------------
ds = BraTSSliceDataset("/home/ubuntu/data/brats20", "train", "flair")
print(f"Dataset size: {len(ds)}\n")

# Color mapping - better colors
BRATS_COLORS = {
    0: (0, 0, 0),           # Background - Black
    1: (255, 0, 0),         # NCR/NET - Red
    2: (0, 255, 0),         # Edema - Green
    3: (0, 0, 255)          # Enhancing - Blue
}

def remap(mask):
    """Remap BraTS labels: 4 -> 3"""
    mask = mask.clone()
    mask[mask == 4] = 3
    return mask

def mask_to_color(mask_np):
    """Convert class mask to RGB image"""
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in BRATS_COLORS.items():
        color[mask_np == k] = v
    return color

# -------------------------------
# Dice Score Calculation
# -------------------------------
def calc_dice(pred, target, num_classes=4):
    """Calculate mean dice across classes"""
    dice_scores = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union > 0:
            dice_scores.append((2.0 * inter / union).item())
    
    return np.mean(dice_scores) if dice_scores else 0.0

# -------------------------------
# Visualization function
# -------------------------------
def visualize_case(idx, save_path=None):
    """Visualize a single case"""
    img, mask = ds[idx]
    mask = remap(mask)

    img_t = img.unsqueeze(0).to(device)

    with torch.no_grad():
        feats = vision_model(img_t)
        logits = seg_head(feats)
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred = torch.argmax(logits, dim=1)[0].cpu()

    # Calculate dice
    dice = calc_dice(pred, mask)

    # Prepare images
    img_np = img[0].numpy()
    gt_color = mask_to_color(mask.numpy())
    pred_color = mask_to_color(pred.numpy())

    # Convert grayscale to RGB for overlay
    img_rgb = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    overlay_gt = cv2.addWeighted(img_rgb, 0.7, gt_color, 0.3, 0)
    overlay_pred = cv2.addWeighted(img_rgb, 0.7, pred_color, 0.3, 0)

    # Plot
    fig = plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 5, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title(f"MRI Slice {idx}", fontsize=12, fontweight='bold')
    plt.axis("off")
    
    plt.subplot(1, 5, 2)
    plt.imshow(gt_color)
    plt.title("Ground Truth", fontsize=12, fontweight='bold')
    plt.axis("off")
    
    plt.subplot(1, 5, 3)
    plt.imshow(pred_color)
    plt.title(f"Prediction\nDice: {dice:.3f}", fontsize=12, fontweight='bold')
    plt.axis("off")
    
    plt.subplot(1, 5, 4)
    plt.imshow(overlay_gt)
    plt.title("GT Overlay", fontsize=12, fontweight='bold')
    plt.axis("off")
    
    plt.subplot(1, 5, 5)
    plt.imshow(overlay_pred)
    plt.title("Pred Overlay", fontsize=12, fontweight='bold')
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None

# -------------------------------
# GIF: Consecutive slices from one patient
# -------------------------------
def create_patient_gif(patient_idx=0, save_path="visuals/patient_animation.gif", stride=5):
    """
    Create GIF animating through slices of one patient
    BraTS has 155 slices per patient
    """
    print(f"\nCreating GIF for patient {patient_idx}...")
    
    start_idx = patient_idx * 155
    end_idx = min(start_idx + 155, len(ds))
    
    frames = []
    temp_dir = "visuals/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    for slice_idx in tqdm(range(start_idx, end_idx, stride), desc="Generating frames"):
        temp_path = f"{temp_dir}/slice_{slice_idx}.png"
        visualize_case(slice_idx, save_path=temp_path)
        frames.append(imageio.imread(temp_path))
    
    # Save GIF
    imageio.mimsave(save_path, frames, fps=3, loop=0)
    
    # Clean up temp files
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"✓ GIF saved: {save_path}")
    return save_path

# -------------------------------
# GIF: Random samples across dataset
# -------------------------------
def create_random_samples_gif(num_samples=20, save_path="visuals/samples_animation.gif"):
    """Create GIF from random samples across dataset"""
    print(f"\nCreating GIF with {num_samples} random samples...")
    
    indices = np.random.choice(len(ds), num_samples, replace=False)
    
    frames = []
    temp_dir = "visuals/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    for idx in tqdm(indices, desc="Generating frames"):
        temp_path = f"{temp_dir}/sample_{idx}.png"
        visualize_case(idx, save_path=temp_path)
        frames.append(imageio.imread(temp_path))
    
    imageio.mimsave(save_path, frames, fps=2, loop=0)
    
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"✓ GIF saved: {save_path}")
    return save_path

# -------------------------------
# Plot training curves
# -------------------------------
def plot_training_curves(log_file="training_log.json", save_path="visuals/training_curves.png"):
    """Plot training metrics"""
    print("\nPlotting training curves...")
    
    if not os.path.exists(log_file):
        print(f"⚠ Log file not found: {log_file}")
        return
    
    with open(log_file) as f:
        log = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(log['epochs'], log['train_loss'], 'b-o', label="Train Loss", linewidth=2, markersize=4)
    axes[0].plot(log['epochs'], log['val_loss'], 'r-o', label="Val Loss", linewidth=2, markersize=4)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Dice plot
    axes[1].plot(log['epochs'], log['val_dice'], 'g-o', linewidth=2, markersize=4)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Dice Score", fontsize=12)
    axes[1].set_title("Validation Dice Score", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Add best dice annotation
    best_idx = np.argmax(log['val_dice'])
    best_dice = log['val_dice'][best_idx]
    best_epoch = log['epochs'][best_idx]
    axes[1].axhline(y=best_dice, color='r', linestyle='--', alpha=0.5)
    axes[1].text(0.02, 0.98, f'Best Dice: {best_dice:.4f}\nEpoch: {best_epoch}',
                 transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Training curves saved: {save_path}")

# -------------------------------
# Evaluate on multiple samples
# -------------------------------
def evaluate_samples(num_samples=50):
    """Calculate dice scores on random samples"""
    print(f"\nEvaluating on {num_samples} random samples...")
    
    indices = np.random.choice(len(ds), num_samples, replace=False)
    dice_scores = []
    
    for idx in tqdm(indices, desc="Evaluating"):
        img, mask = ds[idx]
        mask = remap(mask)
        img_t = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            feats = vision_model(img_t)
            logits = seg_head(feats)
            logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            pred = torch.argmax(logits, dim=1)[0].cpu()
        
        dice = calc_dice(pred, mask)
        dice_scores.append(dice)
    
    print(f"\nResults on {num_samples} samples:")
    print(f"  Mean Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"  Median:    {np.median(dice_scores):.4f}")
    print(f"  Min:       {np.min(dice_scores):.4f}")
    print(f"  Max:       {np.max(dice_scores):.4f}")
    
    return dice_scores

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    os.makedirs("visuals", exist_ok=True)
    
    print("=" * 60)
    print("BraTS Segmentation Visualization Suite")
    print("=" * 60)

    # 1. Visualize individual samples
    print("\n[1/5] Creating individual visualizations...")
    sample_indices = [100, 500, 1000, 1500, 2000]
    for idx in sample_indices:
        visualize_case(idx, save_path=f"visuals/sample_{idx}.png")
        print(f"  ✓ Saved: visuals/sample_{idx}.png")

    # 2. Create patient GIF (consecutive slices)
    print("\n[2/5] Creating patient animation...")
    create_patient_gif(patient_idx=0, save_path="visuals/patient0_animation.gif", stride=5)

    # 3. Create random samples GIF
    print("\n[3/5] Creating random samples animation...")
    create_random_samples_gif(num_samples=20, save_path="visuals/random_samples.gif")

    # 4. Plot training curves
    print("\n[4/5] Plotting training curves...")
    plot_training_curves(save_path="visuals/training_curves.png")

    # 5. Evaluate on test samples
    print("\n[5/5] Evaluating model performance...")
    dice_scores = evaluate_samples(num_samples=100)

    print("\n" + "=" * 60)
    print("✓ All visualizations complete!")
    print("Check the 'visuals' folder for outputs")
    print("=" * 60)
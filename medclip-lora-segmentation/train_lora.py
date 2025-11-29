import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets.brats_dataset import BraTSSliceDataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from medclip import MedCLIPModel, MedCLIPVisionModelViT
import numpy as np
import os

# Disable cuDNN completely to avoid errors
torch.backends.cudnn.enabled = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

############################################################
# 1. SIMPLE SEGMENTATION HEAD
############################################################
class SimpleSegHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=4):
        super().__init__()
        # Simple upsampling decoder
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
        # x shape: (B, C) or (B, C, 1, 1)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        x = self.up1(x)  # (B, 256, 2, 2)
        x = self.up2(x)  # (B, 128, 4, 4)
        x = self.up3(x)  # (B, 64, 8, 8)
        x = self.up4(x)  # (B, 32, 16, 16)
        x = self.final(x)  # (B, num_classes, 16, 16)
        return x

############################################################
# 2. METRICS
############################################################
def dice_score(pred, target, num_classes=4):
    """Multi-class dice coefficient"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    dice_list = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice_list.append((2.0 * intersection / union).item())
    
    return np.mean(dice_list) if dice_list else 0.0

############################################################
# 3. LOAD MODEL
############################################################
print("Loading MedCLIP...")
full_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
full_model.from_pretrained()
vision_model = full_model.vision_model
del full_model
print("✓ MedCLIP loaded\n")

############################################################
# 4. APPLY LORA
############################################################
print("Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)
vision_model = get_peft_model(vision_model, lora_config)
vision_model.to(device)
vision_model.print_trainable_parameters()
print()

# Create segmentation head
seg_head = SimpleSegHead(in_channels=512, num_classes=4).to(device)

############################################################
# 5. DATASET
############################################################
print("Loading dataset...")
dataset = BraTSSliceDataset("/home/ubuntu/data/brats20", "train", "flair")
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)
print(f"Train: {train_size}, Val: {val_size}\n")

############################################################
# 6. OPTIMIZER
############################################################
optimizer = torch.optim.AdamW(
    list(vision_model.parameters()) + list(seg_head.parameters()),
    lr=1e-4
)
criterion = nn.CrossEntropyLoss()

############################################################
# 7. TRAINING LOOP
############################################################
epochs = 40
best_dice = 0.0
patience = 5
wait = 0

os.makedirs("checkpoints", exist_ok=True)

# BraTS label remapping: {0, 1, 2, 4} -> {0, 1, 2, 3}
def remap_labels(mask):
    """Remap BraTS labels to consecutive integers"""
    # BraTS labels: 0 (background), 1 (NCR/NET), 2 (ED), 4 (ET)
    # Remap to: 0, 1, 2, 3
    mask = mask.clone()
    mask[mask == 4] = 3  # Enhancing tumor: 4 -> 3
    return mask

print("Starting training...\n")
print("=" * 60)

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    # TRAIN
    vision_model.train()
    seg_head.train()
    train_loss = 0
    
    for img, mask in tqdm(train_loader, desc="Train"):
        img = img.to(device)
        mask = mask.to(device).long()
        mask = remap_labels(mask)  # Remap BraTS labels
        
        # Forward
        features = vision_model(img)  # (B, 512)
        logits = seg_head(features)   # (B, 4, 16, 16)
        
        # Upsample to match target size
        logits = torch.nn.functional.interpolate(
            logits, size=mask.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        loss = criterion(logits, mask)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vision_model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # VALIDATION
    vision_model.eval()
    seg_head.eval()
    val_loss = 0
    val_dice = 0
    
    with torch.no_grad():
        for img, mask in tqdm(val_loader, desc="Val  "):
            img = img.to(device)
            mask = mask.to(device).long()
            mask = remap_labels(mask)  # Remap BraTS labels
            
            features = vision_model(img)
            logits = seg_head(features)
            logits = torch.nn.functional.interpolate(
                logits, size=mask.shape[-2:],
                mode='bilinear', align_corners=False
            )
            
            loss = criterion(logits, mask)
            val_loss += loss.item()
            
            pred = torch.argmax(logits, dim=1)
            val_dice += dice_score(pred, mask)
    
    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
    
    # Save best model
    if val_dice > best_dice:
        best_dice = val_dice
        wait = 0
        print(f"✓ Best model saved (Dice: {best_dice:.4f})")
        vision_model.save_pretrained("./checkpoints/lora_medclip")
        torch.save(seg_head.state_dict(), "./checkpoints/seg_head.pth")
    else:
        wait += 1
        if wait >= patience:
            print("\nEarly stopping!")
            break

print("\n" + "=" * 60)
print(f"Training complete! Best Dice: {best_dice:.4f}")
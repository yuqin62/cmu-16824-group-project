import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.brats_dataset import BraTSSliceDataset
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from peft import PeftModel
import numpy as np
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

############################################################
# 1. SimpleSegHead (copy from training script)
############################################################
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

############################################################
# 2. Remap labels function
############################################################
def remap_labels(mask):
    mask = mask.clone()
    mask[mask == 4] = 3
    return mask

############################################################
# 3. Dice score
############################################################
def dice_score(pred, target, num_classes=4):
    pred = pred.view(-1)
    target = target.view(-1)
    dice_scores = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        if union > 0:
            dice_scores.append((2 * intersection / union).item())
    return np.mean(dice_scores) if dice_scores else 0.0

############################################################
# 4. Load MedCLIP + LoRA
############################################################
print("\nLoading MedCLIP...")
full_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
full_model.from_pretrained()

vision_model = full_model.vision_model
vision_model = PeftModel.from_pretrained(
    vision_model,
    "./checkpoints/lora_medclip"
)
vision_model.to(device)
vision_model.eval()

############################################################
# 5. Load segmentation head
############################################################
seg_head = SimpleSegHead(in_channels=512, num_classes=4).to(device)
seg_head.load_state_dict(torch.load("checkpoints/seg_head.pth"))
seg_head.eval()

############################################################
# 6. Load validation dataset
############################################################
dataset = BraTSSliceDataset("/home/ubuntu/data/brats20", "train", "flair")
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
_, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

############################################################
# 7. Run validation
############################################################
print("\nRunning validation...")
criterion = nn.CrossEntropyLoss()
val_loss = 0
val_dice = 0

with torch.no_grad():
    for img, mask in val_loader:
        img = img.to(device)
        mask = remap_labels(mask.to(device).long())

        features = vision_model(img)
        logits = seg_head(features)
        logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(logits, mask)
        val_loss += loss.item()

        pred = torch.argmax(logits, dim=1)
        val_dice += dice_score(pred, mask)

val_loss /= len(val_loader)
val_dice /= len(val_loader)

print("\n=== RECOVERED METRICS ===")
print("Recovered Val Loss:", val_loss)
print("Recovered Val Dice:", val_dice)

with open("recovered_metrics.json", "w") as f:
    json.dump({
        "val_loss": val_loss,
        "val_dice": val_dice
    }, f, indent=2)

print("\n✔ Metrics saved to recovered_metrics.json")

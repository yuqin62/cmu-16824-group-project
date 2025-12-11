import os
import sys
import yaml
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# ------------------------------------------------------------------
# Ensure project root is on sys.path so "src" can be imported
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optional: force HuggingFace cache into checkpoints/clip
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "checkpoints", "clip")

from src.models.clip_backbone import load_clip
from dataset.brats_loader import BratsDataset

def load_args(yaml_path):
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
        args = Namespace(**yaml_config)
        return args

def plot_sample_image_and_target(dataset, plot_path):
    '''
    image of shape (C,H,W) torch tensor
    target of shape (H,W) with integer labels, torch tensor
    '''
    image, target = dataset[1]
    image = image.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(15,7))
    axes[0].imshow(image[0,:,:], cmap='gray')
    axes[0].set_title('T1ce')
    axes[1].imshow(image[1,:,:], cmap='gray')
    axes[1].set_title('T2')
    axes[2].imshow(image[2,:,:], cmap='gray')
    axes[2].set_title('FLAIR')

    labels = [0, 1, 2, 4]
    colors = ['black', 'red', 'yellow', 'blue'] # Background, NET/NCR, ED, ET
    cmap = ListedColormap(colors)
    bounds = np.array([0] + labels) - 0.5
    bounds = np.append(bounds, labels[-1] + 0.5)

    bounds = [-0.5, 0.5, 1.5, 3.5, 4.5]

    norm = BoundaryNorm(bounds, cmap.N)
    axes[3].imshow(target, cmap=cmap, norm=norm) 
    axes[3].set_title('Segmentation target')

    
    labels_3class = [1, 2, 4] 
    colors_3class = ['red', 'yellow', 'blue'] # NET/NCR, ED, ET
    cmap_overlay = ListedColormap(colors_3class)
    cmap_overlay.set_under(color='black', alpha=0) 
    bounds_overlay = [0.5, 1.5, 3.5, 4.5] # start at 0.5 to exclude Label 0
    norm_overlay = BoundaryNorm(bounds_overlay, cmap_overlay.N)
    axes[4].imshow(image[2,:,:], cmap='gray')
    axes[4].imshow(target, cmap=cmap_overlay, norm=norm_overlay, alpha=0.6)
    axes[4].set_title('T2 / Segmentation Overlay')
    plt.savefig(plot_path)


def main():
    # Use "cpu" for safety; teammates with GPU can switch to "cuda"
    model, processor = load_clip(device="cpu")
    print("CLIP model and processor loaded successfully.")

    # load dataset yaml
    args_dataset = load_args("../configs/dataset.yaml")
    dataset = BratsDataset(root=args_dataset.train_data_root, modalities=args_dataset.modalities, patch_size=args_dataset.patch_size)
    
    
    plot_sample_image_and_target(dataset, plot_path='./sample_image_target.png')
    print("Sample image and target saved to {plot_path}".format(plot_path='./sample_image_target.png'))

if __name__ == "__main__":
    main()

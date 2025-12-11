import os
import sys
import yaml
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers.image_utils import load_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
import torch

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

    fig, axes = plt.subplots(1, 6, figsize=(15,7))
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
    colors = ['black', 'yellow', 'blue', 'red']  
    # Background label 0,
    # NET/NCR(necrotic) label 1 --> dead cells inside tumor
    # ED(edema) label 2, --> fluid surroudning tumor
    # ET(tumor) label 4 --> active tumor
    
    cmap_overlay = ListedColormap(colors_3class)
    cmap_overlay.set_under(color='black', alpha=0) 
    bounds_overlay = [0.5, 1.5, 3.5, 4.5] # start at 0.5 to exclude Label 0
    norm_overlay = BoundaryNorm(bounds_overlay, cmap_overlay.N)
    axes[4].imshow(image[2,:,:], cmap='gray')
    axes[4].imshow(target, cmap=cmap_overlay, norm=norm_overlay, alpha=0.6)
    axes[4].set_title('T2 / Segmentation Overlay')


    axes[5].imshow(image[0,:,:]+image[1,:,:]+image[2,:,:], cmap='gray')
    axes[5].set_title('Combined Modalities')
    plt.savefig(plot_path)

def split_dataset(dataset, train_ratio=0.7):
    total_size = len(dataset)
    split_index = int(train_ratio * total_size)
    train_indices = list(range(split_index))
    val_indices = list(range(split_index, total_size))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset

def main():
    # Use "cpu" for safety; teammates with GPU can switch to "cuda"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load dataset yaml
    args_dataset = load_args("../configs/dataset.yaml")
    dataset = BratsDataset(root=args_dataset.train_data_root, modalities=args_dataset.modalities, patch_size=args_dataset.patch_size)
    
    TRAIN_RATIO = 0.9  
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=TRAIN_RATIO)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=NUM_WORKERS
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False, 
        num_workers=NUM_WORKERS 
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    # plot_sample_image_and_target(val_dataset, plot_path='./sample_image_target.png')
    print("Sample image and target saved to {plot_path}".format(plot_path='./sample_image_target.png'))

    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)
 
    inference(model, processor, val_dataloader, device, patch_size=args_dataset.patch_size)



def save_prediction(predicted_mask, image_sample, target_sample, batch_idx, prompt_name, patch_size, SAVE_DIR):
    """
    Resizes a single predicted mask, binarizes it, and saves it as a PNG file.
    Args:
        predicted_mask: tensor of shape (H_out, W_out)
        image: tensor of shape (3,H,W)
        target_mask: tensor of shape (H,W)
        prompt_name: descriptive name based on tumor component (e.g., 'ET', 'NC')
    """
    original_h, original_w = patch_size
    
    # resize to (1, H_original, W_original)
    resized_mask_tensor = F.interpolate(
        predicted_mask.float().unsqueeze(0).unsqueeze(0), 
        size=(original_h, original_w),
        mode='nearest' # for binary masks
    ).squeeze()
    
    binary_mask_np = (resized_mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
    
    pil_target = to_pil_image(normalize(target_sample))
    pil_image = to_pil_image(normalize(image_sample))
    pil_combined_image = to_pil_image(normalize(image_sample[0,:,:] + image_sample[1,:,:] + image_sample[2,:,:]))

    pil_mask = Image.fromarray(binary_mask_np * 255) 
    
    
    SAVE_DIR = os.path.join(SAVE_DIR, f"id{batch_idx}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    pil_mask.save(os.path.join(SAVE_DIR, f"mask_{prompt_name}.png"))
    pil_target.save(os.path.join(SAVE_DIR, f"target_grayscale.png"))
    pil_image.save(os.path.join(SAVE_DIR, f"image_all_3_channels.png"))
    pil_combined_image.save(os.path.join(SAVE_DIR, f"image_combined.png"))

    # save individual channels of image
    pil_image_c0 = to_pil_image(normalize(image_sample[0,:,:]))
    pil_image_c1 = to_pil_image(normalize(image_sample[1,:,:]))
    pil_image_c2 = to_pil_image(normalize(image_sample[2,:,:]))
    pil_image_c0.save(os.path.join(SAVE_DIR, f"image_1st_channel.png"))
    pil_image_c1.save(os.path.join(SAVE_DIR, f"image_2nd_channel.png"))
    pil_image_c2.save(os.path.join(SAVE_DIR, f"image_3rd_channel.png"))

    # save target with in color form
    labels = [0, 1, 2, 4]
    colors = ['black', 'yellow', 'blue', 'red'] 
    # Background label 0, NET/NCR(necrotic) label 1, ED(edema) label 2, ET(tumor) label 4
    cmap = ListedColormap(colors)
    bounds = np.array([0] + labels) - 0.5
    bounds = np.append(bounds, labels[-1] + 0.5)
    bounds = [-0.5, 0.5, 1.5, 3.5, 4.5]
    plt.imsave(fname=os.path.join(SAVE_DIR, f"target.png"), arr=target_sample, cmap=cmap, vmin=min(bounds), vmax=max(bounds))

    return torch.from_numpy(binary_mask_np).unsqueeze(0)


def normalize(image):
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val + 1e-5)
    return normalized_image


def compute_dice_score(y_true, y_pred, epsilon=1e-6):
    """Compute dice similarity score for two arrays"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    sum_of_masks = np.sum(y_true_f) + np.sum(y_pred_f)
    dice = (2. * intersection + epsilon) / (sum_of_masks + epsilon)
    return dice

def evaluate_dice_scores(true_mask, predicted_mask, tag):
    """
    Args:
        true_mask: numpy array of shape (H,W) 
        predicted_mask: numpy array of shape (H,W) 
        calculate dice scores for NC (1), ED (2), ET (4)
        Returns a average dice score across the three labels
    """
    labels = {
        "NC": 1,
        "ED": 2,
        "ET": 4
    }
    binary_true_mask = (true_mask == labels[tag]).astype(np.uint8)
    dice = compute_dice_score(binary_true_mask, predicted_mask)
    return dice

    
def inference(model, processor, val_dataloader, device, patch_size):
    SAVE_DIR = "clipseg_inference_results"
    PROMPT_TAGS = {
        "tumor, growth, mass": "ET", # enhancing tumor
        "dead cells in tumor": "NC", # necrotic
        "fluid surrounding tumor": "ED" # edema
    }
    TUMOR_PROMPTS = list(PROMPT_TAGS.keys())
    model.eval()
    all_results = {}
    H,W = patch_size
    total_dice = 0.0
    num_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            images, targets = batch
            targets = targets.to(device)  #(1,H,W)
            B = images.shape[0]
            image_samples = []
            target_samples = []
            pil_images = []
            for i in range(B):
                image_sample = images[i].cpu()
                target_sample = targets[i].cpu()
                
                pil_image = to_pil_image(image_sample)
                pil_images.append(pil_image)
                # for plotting
                image_samples.append(image_sample)
                target_samples.append(target_sample)
                
            dice_per_sample = 0.0
            for tumor_prompt in TUMOR_PROMPTS:
                tag = PROMPT_TAGS[tumor_prompt]
                prompts = [tumor_prompt] * B
                inputs = processor(
                    images=pil_images, 
                    text=prompts,        
                    return_tensors="pt"
                )
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(device)
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(1)
                upsampled_logits = F.interpolate(
                    logits.unsqueeze(1), 
                    size=(H, W), 
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                
                predicted_masks_for_batch = []
                for j in range(B):
                    binary_mask = (upsampled_logits[i] > 0.0).byte().cpu()
                    if binary_mask.shape[0] != H or binary_mask.shape[1] != W:
                        binary_mask = F.interpolate(
                            binary_mask.float().unsqueeze(0).unsqueeze(0), 
                            size=(H, W), 
                            mode='nearest'
                        ).squeeze().byte().cpu()
                    if batch_idx%1000 == 0:
                        save_prediction(
                            binary_mask, 
                            image_samples[j],
                            target_samples[j],
                            batch_idx, 
                            tag,
                            patch_size,
                            SAVE_DIR=SAVE_DIR
                        )
                    dice_score = evaluate_dice_scores(
                        true_mask=target_samples[j].cpu().numpy(),
                        predicted_mask=binary_mask.squeeze(0).cpu().numpy(),
                        tag=tag
                    )
                    dice_per_sample += dice_score
            total_dice += dice_per_sample / len(TUMOR_PROMPTS)
            num_samples += B
    average_dice = total_dice / num_samples
    print("Evaluation completed")
    print(f"Average Dice Score over validation set: {average_dice}")

    with open(os.path.join(SAVE_DIR,"clipseg_dice_scores.txt"), "w") as f:
        f.write(f"Average Dice Score: {average_dice}\n")
           
if __name__ == "__main__":
    main()

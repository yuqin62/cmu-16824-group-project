import os
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from skimage.transform import resize
import glob
import torch.nn.functional as F

class BratsDataset(Dataset):
    """
    Loads BraTS data & returns 2D slices stacked across modalities
    Each case have files with suffixes like *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz, *_seg.nii.gz

    Returns:
        image: (C, H, W) float32 tensor
        target: (H, W) int32 tensor with labels {0, 1, 2, 4}
    """

    def __init__(
        self,
        root: str,
        modalities: List[str] = ("t1", "t1ce", "t2", "flair"),
        patch_size: Tuple[int, int] = (224, 224),
        slice_step=1
    ):

        self.root = root
        self.modalities = list(modalities)
        self.samples = []
        cases = sorted(os.listdir(root))

        for case in cases:
            case_dir = os.path.join(root, case)
            if not os.path.isdir(case_dir):
                continue
            img_paths = []
            for modality in self.modalities:
                img_path = glob.glob(os.path.join(case_dir, f"*_{modality}.nii*"))
                img_paths.append(img_path)
            seg_path = glob.glob(os.path.join(case_dir, "*_seg.nii*"))
    
            if len(img_path) == 0 or len(seg_path) == 0:
                continue
            imgs_modalities = []
            for path in img_paths:
                img = nib.load(path[0])
                img_numpy = img.get_fdata().astype(np.float32)
                imgs_modalities.append(img_numpy)

            img_nii = np.stack(imgs_modalities, 0)
            seg_nii = nib.load(seg_path[0]).get_fdata().astype(np.int32)

            depth = img_nii.shape[3]

            for z in range(0, depth, slice_step):
                img_slice = img_nii[:, :, :, z]
                seg_slice = seg_nii[:, :, z]

                if np.sum(seg_slice) == 0:
                    continue
                self.samples.append((img_paths[0][0], img_paths[1][0], img_paths[2][0], seg_path[0], z))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path_c1, img_path_c2, img_path_c3, seg_path, z = self.samples[idx]

        img_c1_nii = nib.load(img_path_c1)
        img_c2_nii = nib.load(img_path_c2)
        img_c3_nii = nib.load(img_path_c3)

        imgs_modalities = [img_c1_nii.get_fdata(), img_c2_nii.get_fdata(), img_c3_nii.get_fdata()]
        img_nii = np.stack(imgs_modalities, 0).astype(np.float32)
        seg_nii = nib.load(seg_path).get_fdata().astype(np.int16)


        img_slice = img_nii[:, :, :, z]
        seg_slice = seg_nii[:, :, z]
        
        img = (img_slice - np.mean(img_slice)) / (np.std(img_slice) + 1e-5)

        img = torch.from_numpy(img).unsqueeze(0)
        seg = torch.from_numpy(seg_slice).unsqueeze(0).unsqueeze(0)

        img = F.interpolate(img, size=(224, 224), mode="bilinear").squeeze(0)
        seg = F.interpolate(seg.float(), size=(224, 224), mode="nearest").squeeze(0).squeeze(0).long()
        return img, seg

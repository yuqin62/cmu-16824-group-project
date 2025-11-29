import os
import glob
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset

class BraTSSliceDataset(Dataset):
    def __init__(self, root, split="train", modality="flair", slice_step=1):
        self.modality = modality.lower()
        self.samples = []

        if split == "train":
            data_dir = os.path.join(root, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
        else:
            data_dir = os.path.join(root, "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData")

        cases = sorted(os.listdir(data_dir))

        for case in cases:
            case_dir = os.path.join(data_dir, case)
            if not os.path.isdir(case_dir):
                continue

            img_path = glob.glob(os.path.join(case_dir, f"*_{self.modality}.nii*"))
            seg_path = glob.glob(os.path.join(case_dir, "*_seg.nii*"))

            if len(img_path) == 0 or len(seg_path) == 0:
                continue

            img_nii = nib.load(img_path[0])
            seg_nii = nib.load(seg_path[0])

            depth = img_nii.shape[2]

            for z in range(0, depth, slice_step):
                img_slice = img_nii.dataobj[:, :, z]
                seg_slice = seg_nii.dataobj[:, :, z]

                if np.sum(seg_slice) == 0:
                    continue

                self.samples.append((img_path[0], seg_path[0], z))

        print(f"[BraTS] Prepared {len(self.samples)} slice references (lazy load)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, seg_path, z = self.samples[idx]

        img_nii = nib.load(img_path)
        seg_nii = nib.load(seg_path)

        img_slice = np.array(img_nii.dataobj[:, :, z], dtype=np.float32)
        seg_slice = np.array(seg_nii.dataobj[:, :, z], dtype=np.int16)

        img = (img_slice - np.mean(img_slice)) / (np.std(img_slice) + 1e-5)

        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        seg = torch.from_numpy(seg_slice).unsqueeze(0).unsqueeze(0)

        img = F.interpolate(img, size=(224, 224), mode="bilinear").squeeze(0)
        seg = F.interpolate(seg.float(), size=(224, 224), mode="nearest").squeeze(0).squeeze(0).long()

        img = img.repeat(3, 1, 1)
        return img, seg

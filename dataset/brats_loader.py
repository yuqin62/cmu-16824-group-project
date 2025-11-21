import os
import glob
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from skimage.transform import resize


class BratsDataset(Dataset):
    """Loads BRATS cases and returns 2D axial slices stacked across modalities.

    Expects each case to have files with suffixes like: *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz, *_seg.nii.gz
    Dataset returns (C,H,W) image tensor and (H,W) integer mask with values {0,1,2,3}
    """

    def __init__(self, root: str, modalities: List[str] = ["t1", "t1ce","t2","flair"], patch_size=(160,160), slice_axis=2, transform=None, samples_per_case=99999):
        self.root = root
        self.modalities = modalities
        self.patch_size = tuple(patch_size)
        self.slice_axis = slice_axis
        self.transform = transform
        self.samples_per_case = samples_per_case

        self.cases = self._discover_cases()
        self.index_map = self._build_index_map()

    def _discover_cases(self):
        cases = []
        for entry in os.scandir(self.root):
            if not entry.is_dir():
                continue
            # check for modality files
            files = {m: None for m in self.modalities}
            seg = None
            for fname in os.listdir(entry.path):
                for m in self.modalities:
                    if f"_{m}.nii" in fname or f"_{m}.nii.gz" in fname:
                        files[m] = os.path.join(entry.path, fname)
                if "seg" in fname and (fname.endswith('.nii') or fname.endswith('.nii.gz')):
                    seg = os.path.join(entry.path, fname)
            if all(files.values()) and seg:
                cases.append({"dir": entry.path, "mods": files, "seg": seg})
        return cases

    def _build_index_map(self):
        index_map = []  # case_idx
        for i, case in enumerate(self.cases):
            # load one modality to get shape
            img = nib.load(case['mods'][self.modalities[0]]).get_fdata()
            n_slices = img.shape[self.slice_axis]
            for s in range(n_slices):
                index_map.append(i)
                if len(index_map) >= self.samples_per_case:
                    break
        return index_map

    def __len__(self):
        return len(self.index_map)

    def _load_slice(self, filepath: str, slice_idx: int):
        img = nib.load(filepath).get_fdata()
        # select slice along axis
        slc = None
        if self.slice_axis == 0:
            slc = img[slice_idx, :, :]
        elif self.slice_axis == 1:
            slc = img[:, slice_idx, :]
        else:
            slc = img[:, :, slice_idx]
        return slc

    def __getitem__(self, idx):
        case_idx = self.index_map[idx]
        slice_idx = 77
        case = self.cases[case_idx]
        channels = []
        for m in self.modalities:
            arr = self._load_slice(case['mods'][m], slice_idx)
            arr = self._normalize_and_resize(arr)
            channels.append(arr)
        img = np.stack(channels, axis=0).astype(np.float32)

        seg_arr = self._load_slice(case['seg'], slice_idx)
        seg_arr = self._resize_seg(seg_arr)

        # BRATS: labels 1 (necrotic/non-enhancing), 2 (edema), 4 (enhancing) sometimes.
        # background label 0

        img_t = torch.from_numpy(img)
        tgt_t = torch.from_numpy(seg_arr)

        if self.transform:
            img_t, tgt_t = self.transform(img_t, tgt_t)
   
        return img_t, tgt_t

    def _normalize_and_resize(self, arr):
        # simple z-score normalization per-slice
        arr = np.nan_to_num(arr)
        mu = arr.mean()
        sd = arr.std() if arr.std() > 0 else 1.0
        arr = (arr - mu) / sd
        arr = resize(arr, self.patch_size, preserve_range=True, anti_aliasing=True)
        return arr

    def _resize_seg(self, seg):
        # nearest for segmentation
        seg_r = resize(seg, self.patch_size, order=0, preserve_range=True, anti_aliasing=False)
        return seg_r.astype(np.int32)

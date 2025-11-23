import os
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from skimage.transform import resize


class BratsDataset(Dataset):
    """
    Loads BraTS cases and returns 2D slices stacked across modalities.

    Expects each case to have files with suffixes like:
        *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz, *_seg.nii.gz

    Returns:
        image: (C, H, W) float32 tensor
        target: (H, W) int32 tensor with labels {0, 1, 2, 4}
    """

    def __init__(
        self,
        root: str,
        modalities: List[str] = ("t1", "t1ce", "t2", "flair"),
        patch_size: Tuple[int, int] = (160, 160),
        slice_axis: int = 2,
        transform=None,
        min_tumor_fraction: float = 0.0,
    ):
        """
        Args:
            root: root directory containing one subfolder per case.
            modalities: list of modality suffixes to load.
            patch_size: (H, W) to resize slices to.
            slice_axis: which axis to slice along (0, 1, or 2).
            transform: optional callable(img_tensor, mask_tensor) -> (img, mask).
            min_tumor_fraction: if >0, only keep slices where
                (# tumor voxels / # voxels) >= min_tumor_fraction.
                If 0.0, keep any slice with at least one tumor voxel.
        """
        self.root = root
        self.modalities = list(modalities)
        self.patch_size = tuple(patch_size)
        self.slice_axis = slice_axis
        self.transform = transform
        self.min_tumor_fraction = min_tumor_fraction

        self.cases = self._discover_cases()
        self.index_map = self._build_index_map()

    # ------------------------- case discovery ------------------------- #
    def _discover_cases(self):
        cases = []
        for entry in os.scandir(self.root):
            if not entry.is_dir():
                continue

            files = {m: None for m in self.modalities}
            seg = None

            for fname in os.listdir(entry.path):
                fpath = os.path.join(entry.path, fname)

                for m in self.modalities:
                    if f"_{m}.nii" in fname or f"_{m}.nii.gz" in fname:
                        files[m] = fpath

                if "seg" in fname and (fname.endswith(".nii") or fname.endswith(".nii.gz")):
                    seg = fpath

            # keep case only if we have all requested modalities + seg
            if all(files.values()) and seg is not None:
                cases.append({"dir": entry.path, "mods": files, "seg": seg})

        if len(cases) == 0:
            raise RuntimeError(f"No valid BraTS cases found under root: {self.root}")

        return cases

    # ------------------------- index map ------------------------- #
    def _build_index_map(self):
        """
        Build a list of (case_idx, slice_idx) pairs.

        We look at the segmentation volume and keep slices that contain tumor.
        """
        index_map = []

        for case_idx, case in enumerate(self.cases):
            seg_vol = nib.load(case["seg"]).get_fdata()
            seg_vol = np.nan_to_num(seg_vol)

            # shape: (D0, D1, D2)
            n_slices = seg_vol.shape[self.slice_axis]

            for s in range(n_slices):
                if self.slice_axis == 0:
                    seg_slice = seg_vol[s, :, :]
                elif self.slice_axis == 1:
                    seg_slice = seg_vol[:, s, :]
                else:
                    seg_slice = seg_vol[:, :, s]

                # keep slice if it has tumor
                tumor_mask = seg_slice > 0
                frac = tumor_mask.sum() / tumor_mask.size

                if self.min_tumor_fraction == 0.0:
                    if tumor_mask.any():
                        index_map.append((case_idx, s))
                else:
                    if frac >= self.min_tumor_fraction:
                        index_map.append((case_idx, s))

        if len(index_map) == 0:
            raise RuntimeError("Index map is empty. Check min_tumor_fraction / data paths.")

        return index_map

    def __len__(self):
        return len(self.index_map)

    # ------------------------- helpers ------------------------- #
    def _load_slice(self, filepath: str, slice_idx: int):
        vol = nib.load(filepath).get_fdata()
        vol = np.nan_to_num(vol)

        if self.slice_axis == 0:
            slc = vol[slice_idx, :, :]
        elif self.slice_axis == 1:
            slc = vol[:, slice_idx, :]
        else:
            slc = vol[:, :, slice_idx]

        return slc

    def _normalize_and_resize(self, arr: np.ndarray):
        # z-score normalization per slice
        arr = np.nan_to_num(arr)
        mu = arr.mean()
        sd = arr.std() if arr.std() > 0 else 1.0
        arr = (arr - mu) / sd

        arr = resize(
            arr,
            self.patch_size,
            preserve_range=True,
            anti_aliasing=True,
        )
        return arr

    def _resize_seg(self, seg: np.ndarray):
        seg_r = resize(
            seg,
            self.patch_size,
            order=0,               # nearest-neighbor
            preserve_range=True,
            anti_aliasing=False,
        )
        return seg_r.astype(np.int32)

    # ------------------------- main API ------------------------- #
    def __getitem__(self, idx):
        case_idx, slice_idx = self.index_map[idx]
        case = self.cases[case_idx]

        # load and preprocess each modality
        channels = []
        for m in self.modalities:
            arr = self._load_slice(case["mods"][m], slice_idx)
            arr = self._normalize_and_resize(arr)
            channels.append(arr)

        img = np.stack(channels, axis=0).astype(np.float32)

        # load and preprocess segmentation
        seg_arr = self._load_slice(case["seg"], slice_idx)
        seg_arr = self._resize_seg(seg_arr)

        img_t = torch.from_numpy(img)       # (C, H, W)
        tgt_t = torch.from_numpy(seg_arr)   # (H, W)

        if self.transform is not None:
            img_t, tgt_t = self.transform(img_t, tgt_t)

        return img_t, tgt_t

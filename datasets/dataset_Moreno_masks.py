# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import random
import glob
import pickle
import torch
import torch.nn.functional as F
import subprocess


class MorenoMaskDataset(Dataset):
    """
    Dataset of fluorescently labeled cell membranes
    """

    def __init__(
        self,
        data_path,
        # train=False,
        # **kwargs,
    ):
        # self.train = train

        # if not self.train:
        #     self.masks_path = kwargs["masks_path"]
        # else:
        self.data_path = data_path
        self.mask_paths = []
        self.dinoenv = "/work/scratch/yilmaz/miniconda3/envs/dinov2/bin/python"

        # Go through all the subdirectories
        for sub_dir in sorted(glob.glob(self.data_path + "*/Mask/")):
            self.mask_paths.extend(sorted(glob.glob(sub_dir + "*.png")))

    def __len__(self):
        return len(self.mask_paths)

    # def _normalize(self, data):

    #     data -= self.norm1
    #     data /= self.norm2
    #     # minmax data normalization
    #     data = np.clip(data, 1e-5, 1)
    #     return data

    def prepare_mask_and_embedding(self, idx):
        mask_filename = self.mask_paths[idx]
        # mask = io.imread(mask_filename)
        # # Normalize to [0,1]
        # mask = mask / np.max(mask)
        # mask = torch.from_numpy(mask.astype(np.float32))
        # mask = mask[16:80,16:80]
        # mask = torch.squeeze(F.interpolate(mask[None,None,...], [mask.shape[0]*8, mask.shape[1]*8], mode='nearest'))
        # Create embeddings from Dinov2
        mask, emb = self.extract_feats_dinov2(mask_filename)
        # emb = self.extract_feats_dinov2(mask)
        return mask[16:80, 16:80][..., None].repeat(1, 1, 4), emb

    # maybe it's better to extract features in parallel
    def extract_feats_dinov2(self, mask_filename):

        mask = io.imread(mask_filename)

        # now, the mask is the input and the feature map is the conditioning
        # Normalize the mask to [-1,1]
        # input_mask = (mask - min_val) / (max_val - min_val)
        input_mask = (mask - 0.5) * 2.0
        input_mask = torch.from_numpy(input_mask.astype(np.float32))

        # # Normalize to [0,1]
        # cond_mask = mask / np.max(mask)
        # cond_mask = torch.from_numpy(cond_mask.astype(np.float32))
        # cond_mask = cond_mask[16:80,16:80]
        # cond_mask = torch.squeeze(F.interpolate(cond_mask[None,None,...], [cond_mask.shape[0]*8, cond_mask.shape[1]*8], mode='nearest'))##
        # cond_mask = cond_mask[...,None].repeat(1, 1, 3)

        # Construct the full command to activate the Conda environment and run the script
        # full_command = f"conda run -n {self.dinoenv} {command}"
        full_command = [
            "/work/scratch/yilmaz/miniconda3/envs/dinov2/bin/python",
            "datasets/dataset4dinov2.py",
            "--mask_filename",
            mask_filename,
        ]

        # Execute the command
        # result = subprocess.run(full_command, capture_output=True, text=True) #, shell=True
        subprocess.run(full_command, capture_output=True, text=True)  # , shell=True
        with open("datasets/temp_features/temp_feature.pkl", "rb") as f:
            feats = pickle.load(f)
        # feats is a vector of size 1536, reshape it to 512x512x3 as the architecture works only with this shape
        feats = F.pad(feats, (0, 64), "constant", 0)
        feats = torch.reshape(feats, (40, 40))
        feats = torch.squeeze(
            F.interpolate(
                feats[None, None, ...],
                [feats.shape[0] * 13, feats.shape[1] * 13],
                mode="nearest",
            )
        )[:512, :512]
        feats = feats[..., None].repeat(1, 1, 3)
        return input_mask, feats

    def __getitem__(self, idx):

        # keep the prompt empty for now
        prompt = ""
        mask, emb = self.prepare_mask_and_embedding(idx)
        return dict(jpg=mask, txt=prompt, hint=emb)


if __name__ == "__main__":
    dataset = MorenoMaskDataset(
        "/netshares/BiomedicalImageAnalysis/Resources/LiveCellMinerDemo_MorenoBioRxiv/2022_02_16_Data/LSM710/"
    )
    res = dataset.__getitem__(1)
    a = res

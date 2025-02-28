# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import random
import glob
import copy
import torch
import torch.nn.functional as F


class CTCDataset(Dataset):
    """
    Dataset of fluorescently labeled cell nuclei
    """

    def __init__(
        self,
        data_path,
        max_val=36863,
        min_val=32995,
        # train=False,
        # **kwargs,
    ):
        # self.train = train

        # if not self.train:
        #     self.masks_path = kwargs["masks_path"]
        # else:
        self.data_path = data_path
        self.img_paths = []
        self.mask_paths = []
        self.max_val = max_val
        self.min_val = min_val

        # Go through all the subdirectories
        for sub_dir in sorted(glob.glob(self.data_path + "images/*/")):
            self.img_paths.extend(sorted(glob.glob(sub_dir + "*.tif")))
            self.mask_paths.extend(
                sorted(glob.glob(sub_dir.replace("images", "masks") + "*.tif"))
            )

    def __len__(self):
        return len(self.img_paths)

    # def _normalize(self, data):

    #     data -= self.norm1
    #     data /= self.norm2
    #     # minmax data normalization
    #     data = np.clip(data, 1e-5, 1)
    #     return data

    def prepare_img_and_mask(self, idx):
        mask_filename = self.mask_paths[idx]
        img_filename = self.img_paths[idx]

        # Min and max values for images in this dataset
        img = io.imread(img_filename)

        # Normalize to [-1,1]
        img = (img - self.min_val) / (self.max_val - self.min_val)
        img = (img - 0.5) * 2.0
        img = torch.from_numpy(img.astype(np.float32))
        # Upsample the image first because it will be too small when passed
        # thorugh the autoencoder (this is a tentative solution)
        # io.imsave("before_interp.tif", img.numpy())

        # img = torch.squeeze(F.interpolate(img[None,None,...], [img.shape[0]*6, img.shape[1]*6], mode='nearest'))##
        # io.imsave("6_interp.tif", img.numpy())

        mask = io.imread(mask_filename)
        # Normalize to [0,1]
        mask = mask / np.max(mask)
        mask = torch.from_numpy(mask.astype(np.float32))
        # mask = mask[16:80,16:80]##
        # mask = torch.squeeze(F.interpolate(mask[None,None,...], [mask.shape[0]*8, mask.shape[1]*8], mode='nearest'))##
        mask = torch.squeeze(
            F.interpolate(
                mask[None, None, ...],
                [mask.shape[0] * 4, mask.shape[1] * 4],
                mode="nearest",
            )
        )  ##

        # Make RGB
        # to skip the autoencoder, add 4 channels to the input and make it 64x64
        # return img[...,None].repeat(1, 1, 4)[16:80,16:80,:], mask[...,None].repeat(1, 1, 3)##
        img = torch.squeeze(
            F.interpolate(img[None, None, ...], [64, 64], mode="nearest")
        )  ##
        return img[..., None].repeat(1, 1, 4), mask[..., None].repeat(1, 1, 3)  ##

    def __getitem__(self, idx):

        # keep the prompt empty for now
        prompt = ""
        img, mask = self.prepare_img_and_mask(idx)
        return dict(jpg=img, txt=prompt, hint=mask)

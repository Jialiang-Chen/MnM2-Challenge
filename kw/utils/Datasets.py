# -------------------------------------------------------------------------------
# Name:        Datasets.py
# Purpose:     Custom DataSet class for CMRI images and masks
#
# Author:      Kevin Whelan
#
# Created:     24/08/2024
# Copyright:   (c) Kevin Whelan (2024)
# Licence:     MIT
# -------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import os
import math
import numpy as np
import pandas as pd
import nibabel as nib


def apply_normalization(image, normalization_type):
    """
    https://www.statisticshowto.com/normalized/
    :param image:
    :param normalization_type:
    :return:
    """
    if normalization_type == "none":
        return image
    elif normalization_type == "reescale":
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)
        return image
    elif normalization_type == "standardize":
        mean = np.mean(image)
        std = np.std(image)
        image = image - mean
        image = image / std
        return image
    assert False, "Unknown normalization: '{}'".format(normalization_type)


def get_bounding_box(ground_truth_map, nobox, image):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    if nobox:
        bbox = [0, 0, image.size[0], image.size[1]]

    return bbox

class MnMDataset(Dataset):
    """
        Custom Dataset for MnM2 challenge
    """

    def __init__(self, image_dir, mask_dir, nobox=False, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.result = {}
        self.nobox = nobox
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])

        # build corresponding mask filename from image filename
        file_els = self.images[idx].split('_')
        file_els.insert(3, 'gt')
        mask_filename = '_'.join(file_els)
        mask_path = os.path.join(self.mask_dir, mask_filename)
        file_prefix = mask_filename.split('.')[0]

        # load image into numpy array
        image = nib.load(image_path).get_fdata()  # 256 x 256
        # normalize images to [0,255]
        image = apply_normalization(image, 'reescale') * 255
        image = image.astype(np.uint8)

        # load mask into numpy array
        mask = nib.load(mask_path).get_fdata()
        # filter mask to only RV (value = 3.0)
        f_mask = (mask == 3.0)
        # Apply the mask to set all other values to 0
        filtered_mask = np.where(f_mask, mask, 0.0)
        # binarise the mask to {0,1}
        ground_truth_seg = (filtered_mask / 3).astype(np.uint8)
        # multiply by 255 to get in range [0,255]
        ground_truth_seg = ground_truth_seg * 255


        # apply transform
        if self.transform:
            image, ground_truth_seg = self.transform(Image.fromarray(image), Image.fromarray(ground_truth_seg))

        image = np.stack((np.array(image),) * 3, axis=2)  # 255 x 256 x 3

        # compute bounding box
        input_boxes = get_bounding_box(np.array(ground_truth_seg), self.nobox, image)

        self.result["image"] = image
        self.result["mask"] = np.array(ground_truth_seg)
        self.result["filename"] = f'{file_prefix}_pred_mask.png'
        self.result["prompt"] = input_boxes

        return self.result


class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = (np.array(item["mask"]) / 255).astype(int)

        # get bounding box prompt
        prompt = item["prompt"]

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt").to(torch.float32)

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

from datasets.base_dataset import get_transform
from datasets.base_dataset import BaseDataset
from utils.utils import get_nii_files, label_encoder
import torch
import os
import numpy as np
import nibabel as nib

class Segmentation2DDataset(BaseDataset):
    """Represents a 2D segmentation dataset.
    
    Input params:
        configuration: Configuration dictionary.
    """
    def __init__(self, config):
        super().__init__(config)
        # This is probobly really bad but lets roll with it.
        self.img_files = get_nii_files(config['dataset_path'])
        self.img_labels = label_encoder([img_path.rsplit("/",1)[1].split("#",1)[0] for img_path in self.img_files])
        
        
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        
        img = nib.load(img_path)
        x = torch.from_numpy(img.get_fdata())
        y = self.img_labels[idx]
        return x, y

    def __len__(self):
        # return the size of the dataset
        return len(self.img_files)
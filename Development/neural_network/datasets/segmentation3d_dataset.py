from datasets.utils import get_transform,pad_to_shape
from datasets.base_dataset import BaseDataset
from utils.utils import get_nii_files, label_encoder
import torch
import os
import numpy as np
import nibabel as nib
import torch.nn.functional as F

class Segmentation3DDataset(BaseDataset):
    """Represents a 3D segmentation dataset.
    
    Input params:
        configuration: Configuration dictionary.
    """
    def __init__(self, config):
        super().__init__(config)
        # This is probobly really bad but lets roll with it.
        self.img_files = get_nii_files(config['dataset_path'])
        self.img_labels = label_encoder(
            [img_path.rsplit("/",1)[1].split("#",1)[0] for img_path in self.img_files]
        )
        self.augmentation = get_transform(config)
        
    def __getitem__(self, idx): 
        x = nib.load(self.img_files[idx]).get_fdata()
        y = self.img_labels[idx]
        
        # Transform dataset
        x = torch.from_numpy(x).unsqueeze(0).float()
        x = self.augmentation(x)
    
        return x, y

    def __len__(self):
        # return the size of the dataset
        return len(self.img_files)
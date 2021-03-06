from .utils import load_files, filename2labels

import nibabel as nib
from torch.utils.data import Dataset
import torch

class AdniDataset(Dataset):
    def __init__(self, data:list, classes={'CN':0,'MCI':1,'AD':2}, delimiter='_',transform=None):
        self.transform = transform
        self.delimiter=delimiter
        self.classes=classes
        self.labels = filename2labels(data, classes, delimiter)
        self.data = data
        
    def __getitem__(self, idx): 
        x = nib.load(self.data[idx]).get_fdata()
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)
        return x.unsqueeze(0).float(), y

    def __len__(self):
        # return the size of the dataset
        return len(self.data)

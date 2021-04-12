from src.utils import preprocess

import nibabel as nib
from torch.utils.data import Dataset
import torch
from src.utils import preprocess
from src.segmentation.roi_align.roi_align import RoIAlign

class AdniDataset(Dataset):
    def __init__(self, data:list, classes={'CN':0,'MCI':1,'AD':2}, delimiter='_',transform=None):
        super().__init__()
        self.transform = transform
        self.delimiter=delimiter
        self.classes=classes
        self.labels = preprocess.filename2labels(data, classes, delimiter)
        self.data = data
        
    def __getitem__(self, idx): 
        """Load nifti image and convert it to axial view.
        
        Return:
            * Image and label where image has the shape Tuple[C,W,H,D]
        """
        x = preprocess.image2axial(nib.load(self.data[idx]).get_fdata())
        y = self.labels[idx]
        
        x = torch.from_numpy(x)

        if self.transform:
            x = self.transform(x)
        x = preprocess.normalize(x.unsqueeze(0).float()) # Think normalization was missing

        return x, y


    def __len__(self):
        # return the size of the dataset
        return len(self.data)

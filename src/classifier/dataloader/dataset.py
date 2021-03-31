from src.utils import preprocess

import nibabel as nib
from torch.utils.data import Dataset
import torch
from src.utils import preprocess

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
<<<<<<< HEAD
        else:
            x = torch.from_numpy(x)
<<<<<<< HEAD
        return x.unsqueeze(0).float(), y
=======
        x = preprocess.normalize(x.unsqueeze(0).float()) # Think normalization was missing
=======
        #else:
        
        #x = torch.from_numpy(x)
        x = x.unsqueeze(0).float()
        x = preprocess.normalize(x) # Think normalization was missing
>>>>>>> Bug fix
        
        return x, y
>>>>>>> Minor bugfixes to run trainer

    def __len__(self):
        # return the size of the dataset
        return len(self.data)

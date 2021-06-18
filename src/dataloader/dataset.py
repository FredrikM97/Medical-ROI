import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage import filters
from skimage.filters import sobel
from torch.utils.data import Dataset

from src.files import preprocess


class AdniDataset(Dataset):
    """ """
    def __init__(self, data:list, classes={'CN':0,'MCI':1,'AD':2}, transform=None): #, delimiter='_',
        super().__init__()
        self.transform = transform
        #self.delimiter=delimiter
        self.classes=classes
        self.labels = preprocess.folder2labels(data, classes)#, delimiter)
        self.data = data
        
    def __getitem__(self, idx) -> "tuple[tensor,tensor]": 
        """Load nifti image and convert it to axial view.

        Parameters
        ----------
        data : list
            
        classes :
            (Default value = {'CN':0,'MCI':1,'AD':2})
        transform :
            (Default value = None)

        Returns
        -------
        type
            * Image and label where image has the shape Tuple[C,W,H,D]

        
        """
        x = preprocess.image2axial(nib.load(self.data[idx]).get_fdata())
        #mask = x <= 0
        x[x <= 0]=0
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
            #print("Is transform enabled?", x.shape)
        #x = torch.from_numpy(x)
        #x = preprocess.normalize(x.unsqueeze(0).float()) # Think normalization was missing
        x = x.unsqueeze(0).float() # Think normalization was missing

        return x, y


    def __len__(self) -> int:
        """ """
        # return the size of the dataset
        return len(self.data)

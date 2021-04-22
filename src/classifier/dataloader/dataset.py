import nibabel as nib
from torch.utils.data import Dataset
import torch
from src.utils import preprocess
from src.dependencies.roi_align import RoIAlign
from skimage.filters import sobel
from scipy import ndimage as ndi
import numpy as np

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
        # To remove weird background pixels that was supposed to be zero.. Good preprocess
        #elevation_map = sobel(x)
        #mask = ndi.binary_fill_holes(elevation_map)
        #x[mask == 0] = 0
        # Negative values are not allowed so everything below or equal to zero is set to zero!
        mask = x <= 0
        x[mask]=0
        #fft =  np.fft.fftn(x)
        #fft[np.abs(fft) <= 30] = 0

        #back_fft = np.abs(np.fft.ifftn(fft))
        #back_fft[back_fft<0] = 0
        #x = preprocess.normalize(back_fft)

        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
            
        x = torch.from_numpy(x)
        #x = preprocess.normalize(x.unsqueeze(0).float()) # Think normalization was missing
        x = x.unsqueeze(0).float() # Think normalization was missing

        return x, y


    def __len__(self):
        # return the size of the dataset
        return len(self.data)

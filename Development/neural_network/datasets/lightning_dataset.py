from ..utils.dataset import get_nii_files, Kfold, split_data, ClassWeights, get_labels

import torch
import pytorch_lightning as pl 
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize
import torchvision
import itertools
import numpy as np
import sklearn


class LightningDataset(pl.LightningDataModule): 
    delimiter = "_"
    classes = {'CN':0, 'MCI':1,'AD':2}
    
    def __init__(self,dataset_path=None,enable_testset=False,input_shuffle=True, input_seed=0, test_size=0.1,train_params={},val_params={}, img_shape=(95,79),**hparams:dict):
        super().__init__()
        self.trainset,self.valset,self.testset = (np.array([]), np.array([]), np.array([]))
        
        self._datadir = dataset_path
        
        self._trainparams = train_params
        self._valparams = val_params
        
        self._enable_testset=enable_testset
        self._input_shuffle = input_shuffle
        self._files = np.array(get_nii_files(dataset_path))
        self._input_seed = input_seed
        self._test_size = test_size
        self._img_shape = img_shape
        
        self._weights = ClassWeights(delimiter=self.delimiter, classes=self.classes)
        self._kfold = Kfold()
        assert len(self._files) > 0, f"No images found at path: {dataset_path}"
        self._setup()
    
    @property
    def weights(self):
        return self._weights.weights
    
    @property
    def kfold(self):
        return self._kfold
    
    def _setup(self): 
        
        if self._input_shuffle:
            self._files = sklearn.utils.shuffle(self._files, random_state=self._input_seed)
            
        if self._enable_testset: 
            self.trainset, self.testset = split_data(self._files, test_size=self.test_size)
        else:
            self.trainset = self._files
        
        self.trainset, self.valset = split_data(self.trainset)

        print(f"Dataset sizes - Training: {len(self.trainset)} Validation: {len(self.valset)} Test: {len(self.testset)}")
        
    def train_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self._img_shape)
        ])
        dataloader = DataLoader(NiiDataset(self.trainset, self.classes, transform=transform),
                                batch_size=self._trainparams['batch_size'],
                                shuffle=True,
                                num_workers=self._trainparams['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self._img_shape)
        ])
        dataloader = DataLoader(NiiDataset(self.valset, self.classes, transform=transform),
                                batch_size=self._valparams['batch_size'],
                                shuffle=False,
                                num_workers=self._valparams['num_workers'])

        return dataloader
    
class NiiDataset(Dataset):
    #_delimiter = None
    #_classes = None
    
    def __init__(self, data:list, classes:dict, delimiter:str='_',transform=None):
        
        self._classes=classes
        self.transform = transform
        self._delimiter=delimiter
        self.labels = get_labels(data, classes, delimiter)
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

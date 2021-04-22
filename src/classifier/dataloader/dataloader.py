
import pytorch_lightning as pl 
from torch.utils.data import DataLoader
import torch
import torchvision
#from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import load
from .dataset import AdniDataset
from src.segmentation import RoiTransform
from src import BASEDIR
#from .weights import ClassWeights
#from .kfold import Kfold
from src.utils import preprocess



class AdniDataloader(pl.LightningDataModule): 
    def __init__(self,data_dir, seed=0, batch_size=6, shuffle=True, validation_split=0.1, num_workers=1, img_shape=(79,95,79),**hparams:dict):
        super().__init__()
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed
        self.img_shape = img_shape
        self.dataset = load.load_files(BASEDIR + "/"+data_dir)
        
        self.trainset, self.valset = self._split(self.validation_split)

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        
        print(f"Data directory: {data_dir}\nDataset sizes - Training: {len(self.trainset)} Validation: {len(self.valset)}")
        

    def train_dataloader(self):
        transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(self.img_shape)
            torchvision.transforms.Lambda(lambda images: preprocess.preprocess_image(images,input_shape=self.img_shape))
        ])
        return DataLoader(AdniDataset(self.trainset,transform=transform),
                                shuffle=True,
                                **self.init_kwargs)
    
    def val_dataloader(self):
        transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(self.img_shape)
            torchvision.transforms.Lambda(lambda images: preprocess.preprocess_image(images,input_shape=self.img_shape))
        ])
        return DataLoader(AdniDataset(self.valset, transform=transform),
                                shuffle=False,
                                **self.init_kwargs)

    def _split(self, split):
        if split == 0.0:
            return self.dataset, np.array([])
        
        train_samples, valid_samples = train_test_split(self.dataset, test_size=split, random_state=self.seed, shuffle=self.shuffle)
        return train_samples, valid_samples

class RoiDataloader(pl.LightningDataModule): 
    def __init__(self,data_dir, seed=0, batch_size=6, shuffle=True, validation_split=0.1, num_workers=1, img_shape=(79,95,79), roi_hparams={'output_shape':None, 'boundary_boxes':[]}, **hparams:dict):
        super().__init__()
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed
        self.img_shape = img_shape
        
        self.dataset = load.load_files(BASEDIR + "/"+data_dir)
        self.trainset, self.valset = self._split(self.validation_split)
       
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        
        print(f"Dataset sizes - Training: {len(self.trainset)} Validation: {len(self.valset)}")
        

    def train_dataloader(self):
        transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(self.img_shape),
            torchvision.transforms.Lambda(lambda images: preprocess.preprocess_image(images,input_shape=self.img_shape))
        ])
        return DataLoader(AdniDataset(self.trainset,transform=transform),
                                shuffle=True,
                                **self.init_kwargs)
    
    def val_dataloader(self):
        transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(self.img_shape),
            torchvision.transforms.Lambda(lambda images: preprocess.preprocess_image(images,input_shape=self.img_shape))
        ])
        return DataLoader(AdniDataset(self.valset, transform=transform),
                                shuffle=False,
                                **self.init_kwargs)

    def _split(self, split):
        if split == 0.0:
            return self.dataset, np.array([])
        
        train_samples, valid_samples = train_test_split(self.dataset, test_size=split, random_state=self.seed, shuffle=self.shuffle)
        return train_samples, valid_samples

class ToDevice(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return sample.to(self.device)
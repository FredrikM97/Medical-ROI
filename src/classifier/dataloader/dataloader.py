
import pytorch_lightning as pl 
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from src import BASEDIR
from src.utils import load
from src.utils import preprocess
from .dataset import AdniDataset
from .augmentation import augment, randomNoise, randRange

import torchvision.transforms as tf
import random

class AdniDataloader(pl.LightningDataModule): 
    def __init__(self,data_dir, batch_size=6, shuffle=True, num_workers=1, img_shape=(79,95,79), classes={},**hparams:dict):
        super().__init__()
        self.shuffle = shuffle
        self.seed = hparams.get('seed',0)
        self.split_conf = hparams.get('split',{})
        self.img_shape = img_shape
        self.data_dir = data_dir
        self.classes = classes
        self.augmentation = hparams.get("augmentation", None)
        self.kfold = None
        self.kfold_index = 0
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        use_augmentation = [
            tf.Lambda(lambda images: torch.from_numpy(images)),
            tf.RandomAffine(
                degrees=(0, 180), 
                translate=(0.001, 0.001),
           
            )
        
        ] if self.augmentation['enable'] else []
        
        self.train_transform = torchvision.transforms.Compose([
            *use_augmentation,
            tf.Lambda(lambda images: preprocess.preprocess_image(images,input_shape=self.img_shape)),
            tf.Lambda(lambda images: torch.from_numpy(images))
        ])
        
        self.test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda images: torch.from_numpy(preprocess.preprocess_image(images,input_shape=self.img_shape)))
            
        ])
        
        
        # Init kfold if the config require it.
        if self.split_conf['kfold_enable']:
            kfold = StratifiedKFold(self.split_conf['folds'],shuffle=True, random_state=self.seed) 
        
            # Load datafiles 
            dataset_full = load.load_files(BASEDIR + "/"+self.data_dir)
            labels = preprocess.folder2labels(dataset_full, self.classes)#, self.delimiter)
        
            self.kfold = kfold.split(dataset_full,labels)
            self.next_fold()
        
        
    def setup(self, stage=None):
        dataset_full = load.load_files(BASEDIR + "/"+self.data_dir)
        
        # Assign kfold or split depending on the configuration
        if self.kfold: 
            dataset_splitted = [torch.utils.data.Subset(dataset_full, idxs) for idxs in self.folds]
        else:
            dataset_splitted = _split(dataset_full, test_size=self.split_conf['val_size'], random_state=self.seed, shuffle=self.shuffle)
        
        self.dataset_splitted = dataset_splitted
        
        self.adni_train, self.adni_val = [ #,delimiter=self.delimiter
            AdniDataset(data, transform=transform, classes=self.classes) for transform, data in zip([self.train_transform,self.test_transform], dataset_splitted)
            ]
        
    def __str__(self):
        return (
            f"***Defined dataloader:***\n"
            f"Data directory: {self.data_dir}\n"
            f"Dataset sizes - Training: {len(self.adni_train)} Validation: {len(self.adni_val)}\n"
            f"Seed: {self.seed}\n"
            f"Augmentation: {'Enabled' if self.augmentation['enable'] else 'Disabled'}\n"
            f"KFold: {'Enabled - Fold: ' + str(self.kfold_index) + '/' + str(self.split_conf['folds']) if self.split_conf['kfold_enable'] else 'Disabled'}\n")
        
    
    def next_fold(self):
        if self.split_conf['folds'] <= self.kfold_index: return False
        self.kfold_index +=1
        self.folds = next(self.kfold)
        return True
    
    def train_dataloader(self):
        return DataLoader(self.adni_train,
                        shuffle=True,
                        **self.init_kwargs
                     )
    
    def val_dataloader(self):
        return DataLoader(self.adni_val,
                        shuffle=False,
                        **self.init_kwargs
                    )
    
class ToDevice(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return sample.to(self.device)
    
def _split(dataset, test_size, random_state=0, shuffle=False):
        if test_size == 0.0:
            return dataset, np.array([])
        
        train_samples, valid_samples = train_test_split(dataset, test_size=test_size, random_state=random_state, shuffle=shuffle)
        return train_samples, valid_samples
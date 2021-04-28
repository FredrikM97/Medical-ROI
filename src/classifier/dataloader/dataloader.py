
import pytorch_lightning as pl 
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from src import BASEDIR
from src.utils import load
from src.segmentation import RoiTransform
from src.utils import preprocess
from .dataset import AdniDataset
from .augmentation import augment

class AdniDataloader(pl.LightningDataModule): 
    def __init__(self,data_dir, seed=0, batch_size=6, shuffle=True, num_workers=1, img_shape=(79,95,79), classes={},delimiter='_',split={},**hparams:dict):
        super().__init__()
        self.shuffle = shuffle
        self.seed = seed
        self.img_shape = img_shape
        self.split_conf = split
        self.data_dir = data_dir
        self.delimiter = delimiter
        self.classes = classes
        self.augmentation = hparams.get("augmentation", None)
        self.kfold = None
        self.kfold_index = 0
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }
    
        self.train_transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(self.img_shape)
            torchvision.transforms.Lambda(lambda images: preprocess.preprocess_image(images,input_shape=self.img_shape)),
            torchvision.transforms.Lambda(lambda images: torch.from_numpy(augment(images,input_shape=self.img_shape))) if self.augmentation,
            torchvision.transforms.Lambda(lambda images: torch.from_numpy(images)),

        ])
        
        self.test_transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(self.img_shape)
            torchvision.transforms.Lambda(lambda images: torch.from_numpy(preprocess.preprocess_image(images,input_shape=self.img_shape)))
            
        ])
        
        
        # Init kfold if the config require it.
        if self.split_conf['kfold_enable']:
            kfold = StratifiedKFold(self.split_conf['folds'],shuffle=True, random_state=None) 
        
            # Load datafiles 
            dataset_full = load.load_files(BASEDIR + "/"+self.data_dir)
            labels = preprocess.filename2labels(dataset_full, self.classes, self.delimiter)
        
            self.kfold = kfold.split(dataset_full,labels)
            self.next_fold()
        
        
    def setup(self, stage=None):
        print("SETUUP",stage)
        dataset_full = load.load_files(BASEDIR + "/"+self.data_dir)
        
        # Assign kfold or split depending on the configuration
        if self.kfold: 
            dataset_splitted = [torch.utils.data.Subset(dataset_full, idxs) for idxs in self.folds]
        else:
            dataset_splitted = _split(dataset_full, test_size=self.split_conf['val_size'], random_state=self.seed, shuffle=self.shuffle)
        
        if stage =='fit':
            self.adni_train, self.adni_val = [AdniDataset(data, transform=self.train_transform,delimiter=self.delimiter, classes=self.classes) for data in dataset_splitted]
        else:
            self.adni_train, self.adni_val = [AdniDataset(data, transform=self.test_transform,delimiter=self.delimiter, classes=self.classes) for data in dataset_splitted]
            
        # Info of the dataset
        print((
            f"***Defined dataloader:***\n"
            f"Data directory: {self.data_dir}\n"
            f"Dataset sizes - Training: {len(self.adni_train)} Validation: {len(self.adni_val)}\n"
            f"KFold: {'Enabled - Fold: ' + str(self.kfold_index) + '/' + str(self.split_conf['folds']) if self.split_conf['kfold_enable'] else 'Disabled'}\n"
        ))
    
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
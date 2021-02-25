from utils import get_nii_files, label_encoder

import torch
import pytorch_lightning as pl 
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
#https://www.kaggle.com/lezwon/parallel-kfold-training-on-tpu-using-pytorch-li
#https://forums.pytorchlightning.ai/t/kfold-vs-monte-carlo-cross-validation/587
import itertools
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

class LightningDataset(pl.LightningDataModule): 
    trainset,testset,valset = (0,0,0)
    def __init__(self,dataset_path=None,enable_testset=False,input_shuffle=True, input_seed=0, test_size=0.1,train_params={},val_params={},**hparams:dict):
        super().__init__()
        self._hparams = hparams
        self.dataset_path = dataset_path
        self.train_params = train_params
        self.val_params = val_params
        self.enable_testset=enable_testset
        self.input_shuffle = input_shuffle
        self.img_files = get_nii_files(dataset_path)
        self.input_seed = input_seed
        self.test_size = test_size
        
        assert len(self.img_files) > 0, f"No images found at path: {dataset_path}"
        self._setup()
    
    def _setup(self): 
        #dataset = NiiDataset(self.img_files)
        
        if self.input_shuffle:
            self.img_files = sklearn.utils.shuffle(self.img_files, random_state=self.seed)
            
        if self.enable_testset: 
            self.trainset, self.testset = split_data(self.img_files, test_size=self.test_size)
        else:
            self.trainset = self.img_files
            
            
        #self.get_current_fold = 0
        #self.kfold_splits = kfold(dataset, n_splits=self._hparams['fold_splits'], shuffle=True)
        #self._next_fold()
        self.trainset, self.valset = split_data(self.trainset)

        print(f"Dataset sizes - Training: {len(self.trainset)} Validation: {len(self.valset)} Test: {len(self.testset)}")
    
    def train_dataloader(self):
        #assert len(self.train_set) % self._hparams['train_params']['batch_size'] != 1, "A batch of size 1 is not allowed!"

        dataloader = DataLoader(self.trainset,
                                batch_size=self.train_params['batch_size'],
                                shuffle=True,
                                num_workers=self.train_params['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        #assert len(self.val_set) % self._hparams['train_params']['batch_size'] != 1, "A batch of size 1 is not allowed!"
        #dataset = NiiDataset(train=False, data_dir=self.img_files)
        
        dataloader = DataLoader(self.valset,
                                batch_size=self.val_params['batch_size'],
                                shuffle=False,
                                num_workers=self.val_params['num_workers'])

        return dataloader
    
class NiiDataset(Dataset):
    def __init__(self, data: list):
        
        self.labels = label_encoder(
            [img_path.rsplit("/",1)[1].split("#",1)[0] for img_path in data]
        )
        self.data = data

    def __getitem__(self, idx): 
        x = nib.load(self.data[idx]).get_fdata()
        y = self.labels[idx]
        
        x = torch.from_numpy(x).unsqueeze(0).float()
        return x, y

    def __len__(self):
        # return the size of the dataset
        return len(self.data)

class ClassWeights:
    unique_classes = None
    labels = None
    weights = None
    delimiter = '#'
    def __init__(self, classes:dict):
        self.unique_classes=classes
        
    def compute(self, images):
        self.labels = np.numpy([self.classes[img_path.rsplit("/",1)[1].split(self.delimiter,1)[0]] for img_path in data]) # Modified for this dataset..
        
        # Compute weights
        self.weights = sklearn.utils.class_weight.compute_class_weight('balanced', classes=list(self.unique_classes.keys()), y=labels)
    
    def get_tensor(self):
        return torch.from_numpy(self.weights).float()#.cuda()
    
    def get(self):
        return self.weights
        
class Kfold:
    
    def __init__(self, dataset, n_splits=5, shuffle=False, random_state=None):
        self._trainData = None
        self._valData = None
        self._fold_idx = None
        self._n_splits = None
        self.shuffle=None
        self.random_state = random_state
        self.kfold(dataset, n_splits, shuffle=shuffle, random_state=random_state)
        
    def kfold(self,dataset, n_splits, shuffle=False, random_state=None):
        self.shuffle=shuffle
        self._n_splits = n_splits
        idxs = KFold(n_splits, shuffle=shuffle, random_state=random_state).split(np.arange(len(dataset)))
        self._set(((torch.utils.data.Subset(dataset, train_idxs), torch.utils.data.Subset(dataset, val_idxs)) for train_idxs, val_idxs in idxs))
        self.next()
    
    def _set(self,folds):
        self._fold_idx = 1
        self._folds = folds
        
    def get(self):
        return self._trainData, self._valData
    
    def index(self) -> int:
        return self._fold_idx
    
    def size(self):
        return self._n_splits
    
    def exists(self) -> bool:
        return self.index() < self.size()
        
    def next(self) -> None:
        assert self.exists(), "Cant find more folds!"
        self._fold_idx += 1
        self._trainData, self._valData = next(self._folds)
    
def split_data(data,test_size=0.1, random_state=None, shuffle=False):
    return train_test_split(data, test_size=test_size, random_state=random_state, shuffle=shuffle)
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

class LightningDataset(pl.LightningDataModule): 
    
    def __init__(self,**hparams:dict):
        super().__init__()
        self._hparams = hparams
        self.img_files = get_nii_files(self._hparams['dataset_path'])
        assert len(self.img_files) > 0, f"No images found at path: {self._hparams['dataset_path']}"
    
    def setup(self,stage=None):
        dataset = NiiDataset(self.img_files)
        
        if self._hparams['val_type'] == 'kfold':
            print("Going with kfold!")
            self.kfold_splits = kfold(dataset, n_splits=self._hparams['fold_splits'])
            self.train_set, self.val_set = next(self.kfold_splits)
        else:
            print("Going with default split!")
            # If not kfold then do something else...
            num_train_samples = len(dataset)*0.9
            self.train_set, self.val_set = (dataset[:num_train_samples], dataset[num_train_samples:])

    def _on_epoch_end(self):
        if self._hparams['val_type'] == 'kfold':
            self.train_set, self.val_set = next(self.trainer.datamodule.kfold_splits)
         
    def train_dataloader(self):
        #assert len(self.train_set) % self._hparams['train_params']['batch_size'] != 1, "A batch of size 1 is not allowed!"

        dataloader = DataLoader(self.train_set,
                                batch_size=self._hparams['train_params']['batch_size'],
                                shuffle=True,
                                num_workers=self._hparams['train_params']['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        #assert len(self.val_set) % self._hparams['train_params']['batch_size'] != 1, "A batch of size 1 is not allowed!"
        #dataset = NiiDataset(train=False, data_dir=self.img_files)
        
        dataloader = DataLoader(self.val_set,
                                batch_size=self._hparams['val_params']['batch_size'],
                                shuffle=False,
                                num_workers=self._hparams['val_params']['num_workers'])

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

def kfold(dataset, n_splits=5):
    idxs = itertools.cycle(KFold(n_splits).split(np.arange(len(dataset))))
    for train_idxs, val_idxs in idxs:
        yield torch.utils.data.Subset(dataset, train_idxs), torch.utils.data.Subset(dataset, val_idxs)
        
from utils import get_nii_files, label_encoder

import torch
import pytorch_lightning as pl 
import nibabel as nib
from torch.utils.data import DataLoader, Dataset

class LightningDataset(pl.LightningModule): 
    
    def __init__(self, hparams):
        super().__init__()
        
        self.img_files = get_nii_files(hparams['dataset_path'])
        assert len(self.img_files) > 0, f"No images found at path: {hparams['dataset_path']}"
        
        self.dims = hparams['input_size']
        self.save_hyperparameters()
        
    def train_dataloader(self):
        dataset = NiiDataset(train=True, data_dir=self.img_files)

        dataloader = DataLoader(dataset,
                                batch_size=self.hparams['hparams']['train_params']['batch_size'],
                                shuffle=True,
                                num_workers=self.hparams['hparams']['train_params']['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        dataset = NiiDataset(train=False, data_dir=self.img_files)
        
        dataloader = DataLoader(dataset,
                                batch_size=self.hparams['hparams']['val_params']['batch_size'],
                                shuffle=False,
                                num_workers=self.hparams['hparams']['val_params']['num_workers'])

        return dataloader
    
class NiiDataset(Dataset):
    def __init__(self, train: bool, data_dir: list):
        
        # TODO: Fix this. If data not shuffle it will always prio data in the bottom!
        num_train_samples = int(len(data_dir) * 0.9)
        
        if train:
            data = data_dir[:num_train_samples]
        else:
            data = data_dir[num_train_samples:]
        
        self.labels = label_encoder(
            [img_path.rsplit("/",1)[1].split("#",1)[0] for img_path in data]
        )
        self.data = data
        self.data_dir = data_dir

    def __getitem__(self, idx): 
        x = nib.load(self.data[idx]).get_fdata()
        y = self.labels[idx]
        
        x = torch.from_numpy(x).unsqueeze(0).float()
        return x, y

    def __len__(self):
        # return the size of the dataset
        return len(self.data)
    
#import rising.transforms as rtr
import torch
#from rising.loading import DataLoader
import pytorch_lightning as pl 
#from rising.loading import Dataset
from utils import get_nii_files, label_encoder
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
class LightningDataset(pl.LightningModule): 
    
    def __init__(self, hparams):
        super().__init__()
        
        self.img_files = get_nii_files(hparams['dataset_path'])
        self.dims = hparams['input_size']
        self.save_hyperparameters()
        
    def train_dataloader(self):
        dataset = NiiDataset(train=True, data_dir=self.img_files)
        
        #batch_transforms = rtr.Compose([
            #rtr.Lambda(lambda img: img.unsqueeze(0).float())
        #])
        print("asdas",self.hparams['hparams']['train_params']['batch_size'])
        dataloader = DataLoader(dataset,
                                batch_size=self.hparams['hparams']['train_params']['batch_size'],
                                #batch_transforms=batch_transforms,
                                shuffle=True,
                                #sample_transforms=common_per_sample_trafos(),
                                #pseudo_batch_dim=True,
                                num_workers=self.hparams['hparams']['train_params']['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        dataset = NiiDataset(train=False, data_dir=self.img_files)
        
        #batch_transforms = rtr.Compose([
            #rtr.Lambda(lambda img: (img['data'].unsqueeze(0).float(), img['label']))
        #])
        
        dataloader = DataLoader(dataset,
                                batch_size=self.hparams['hparams']['val_params']['batch_size'],
                                #batch_transforms=batch_transforms,
                                shuffle=False,
                                #sample_transforms=common_per_sample_trafos(),
                                #pseudo_batch_dim=True,
                                num_workers=self.hparams['hparams']['val_params']['num_workers'])

        return dataloader
    
class NiiDataset(Dataset):
    def __init__(self, train: bool, data_dir: list):
        
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
    
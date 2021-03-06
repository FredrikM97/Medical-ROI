from .dataset import AdniDataset
import pytorch_lightning as pl 
from torch.utils.data import DataLoader
from skimage.transform import resize
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split

class AdniDataModule(plt.LightningDataModule): 
    def __init__(self,data_dir, seed=0, batch_size=6, shuffle=True, validation_split=0.1, num_workers=1, img_shape=(95,79),**hparams:dict):
        super().__init__()
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed
        self.img_shape = img_shape
        self.dataset = load_files(data_dir)
        
        self.trainset, self.validset = self._split(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'num_workers': num_workers
        }
     
    def train_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_shape)
        ])
        dataloader = DataLoader(dataset(self.dataset[self.train_sampler],transform=transform),
                                shuffle=True,
                                **self.init_kwargs)
        return dataloader
    
    def val_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_shape)
        ])
        dataloader = DataLoader(dataset(self.dataset[self.valid_sampler], transform=transform),
                                shuffle=False,
                                **self.init_kwargs)

    def _split(self, split):
        if split == 0.0:
            return self.dataset, np.array([])
        
        train_samples, valid_samples train_test_split(self.dataset, test_size=split, random_state=self.seed, shuffle=self.shuffle)
    
        return train_samples, valid_samples

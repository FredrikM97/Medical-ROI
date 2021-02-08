from configs import load_config
from models import create_model
from datasets import create_dataset

from pytorch_lightning.callbacks import ModelCheckpoint, progress
import pytorch_lightning as pl
#from pytorch_lightning import loggers as pl_loggers
#from pytorch_lightning.callbacks.progress import ProgressBarBase

import sys
import torch

class Agent:
    def __init__(self, config_name:str, export:bool=True):
        print('Setup configurations...')
        # ****** Setup configurations ******
        self.config = load_config(config_name)
        
        
        # ****** Setup loggers ******
       
        checkpoint_callback = ModelCheckpoint(
            save_top_k=0, # disable?
            filename=self.config['logs']['checkpoint']+'{epoch}-{val_loss:.2f}',
            #monitor='val_loss', 
            mode='min', 
        )
        #progressbar_callback = ProgressBar()
        progressbar_callback = LitProgressBar()
        
        tb_logger = pl.loggers.TensorBoardLogger(self.config['logs']['tensorboard'], name=self.config['model_params']['model_name'])
        
        # ****** Setup model ******
        self.model = create_model(self.config['model_params'])
        
        # ****** Setup dataloader ******
        dataset = create_dataset(self.config['dataset_params'])
        
        self.train_loader = dataset.train_dataloader()
        self.val_loader = dataset.val_dataloader()
        
        print(f"Dataset size: Train: {len(self.train_loader)} Val: {len(self.val_loader)}")
        
        # ****** Check if gpu exists ******
        if torch.cuda.is_available():
            gpu_availible = self.config['gpus']

        else:
            gpu_availible = None
        # ****** Setup trainer ******
        
        self.trainer = pl.Trainer(
            max_epochs=self.config['model_params']['max_epochs'], 
            profiler=None, 
            reload_dataloaders_every_epoch=self.config['reload_dataloaders_every_epoch'],
            gpus=gpu_availible, 
            logger=tb_logger,
            callbacks=[checkpoint_callback,progressbar_callback],
            progress_bar_refresh_rate=self.config['progress_bar_refresh_rate']
        )
        
    def fit(self):
        self.trainer.fit(
            self.model, 
            train_dataloader=self.train_loader, 
            val_dataloaders=self.val_loader
        )
    
    def test(self):
        trainer.test(
            test_dataloaders=self.test_dataloader
        )
        

class LitProgressBar(progress.ProgressBarBase):
#https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/progress.html#ProgressBarBase.on_validation_batch_end
    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True
        self.train_content = ''
        self.val_content = ''
        

    def disable(self):
        self.enable = False
    
    def on_epoch_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        print()
        print("",end="", flush=True)
        
    #def on_validation_start(self, trainer, pl_module):
    #    super().on_train_start(trainer, pl_module)
    #    pass

        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs,batch, batch_idx, dataloader_idx) 
        
        del trainer.progress_bar_dict['v_num']
        
        self.train_content = f'Epoch {trainer.current_epoch+1} [{batch_idx+1:.00f}/{self.total_train_batches:.00f}] {trainer.progress_bar_dict}'
        
        self._update()
        
    #def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #    super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) 
    #    
    #    self.val_content = f'Epoch {trainer.current_epoch+1} [{batch_idx:.00f}/{self.total_train_batches:.00f}] {str(trainer.progress_bar_dict)[1:-1]}'
    #    self._update()
        
    def _update(self):
        print(self.train_content + self.val_content, end="\r", flush=True)
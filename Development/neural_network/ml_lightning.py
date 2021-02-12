from configs import load_config
from models import create_model
from datasets import create_dataset
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.callbacks import ModelCheckpoint, progress
import pytorch_lightning as pl

import sys
import torch
import numpy as np

class Agent:
    def __init__(self, config_name:str, export:bool=True):
        print('Setup configurations...')
        # ****** Setup configurations ******
        self.config = load_config(config_name)
        
        # ****** Setup seed *******
        #np.random.seed(self.config['agent']['seed'])
        
        # ****** Setup dataloader ******
        self.dataset = create_dataset(**self.config['dataset_params'])
        
        # ****** Setup model ******
        self.model = create_model(**self.config['model_params'], class_weights=self.dataset._get_class_weights())
        
        
        
    def setup_trainer(self):
        self.trainer = pl.Trainer(
            max_epochs=self.config['model_params']['max_epochs'], 
            profiler=None, 
            reload_dataloaders_every_epoch=self.config['reload_dataloaders_every_epoch'],
            gpus=self.get_gpu(), 
            logger=self.logger(),
            callbacks=self.callbacks(),
            progress_bar_refresh_rate=self.config['progress_bar_refresh_rate'],
            num_sanity_val_steps=self.config['num_sanity_val_steps'],
            benchmark=True,
            auto_lr_find=True,
            
        )
        
    def logger(self):
        return pl.loggers.TensorBoardLogger(self.config['logs']['tensorboard'], name=self.config['model_params']['model_name'])
     
    def callbacks(self) -> list:
        lit_progress = LitProgressBar()
        checkpoint_callback = ModelCheckpoint(
            save_top_k=0, # disable?
            filename=self.config['logs']['checkpoint']+'{epoch}-{val_loss:.2f}',
            mode='min', 
        )
        return [lit_progress, checkpoint_callback]
    
    def get_gpu(self):
        return -1 if torch.cuda.is_available() else None
    
    def fit(self, cv=False) -> None:
        self.setup_trainer()
        if self.config['agent']['kfold']:
            self.__fit_cv()
        else:
            self.__fit()
    
    def __fit(self):
        self.setup_trainer()
        self.trainer.fit(
            self.model, 
            datamodule=self.dataset
        )
    
    def __fit_cv(self) -> None:
        print("Fitting with cv")
         
        while self.dataset._has_folds():
            # call fit
            print(f"Validation on fold: {self.dataset._get_fold()}")
            self.__fit()
            if self.trainer._state == TrainerState.INTERRUPTED: break

            # store metrics
            # get next fold
            self.dataset._next_fold()
 
    def get_info():
        pass
        

class LitProgressBar(progress.ProgressBarBase):
#https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/progress.html#ProgressBarBase.on_validation_batch_end
    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable()

    def disable(self):
        self._enable = False
    
    def enable(self):
        self._enable = True
        
    def on_epoch_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        print()
        print("",end="", flush=True)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs,batch, batch_idx, dataloader_idx) 
        
        con = f'Epoch {trainer.current_epoch+1} [{batch_idx+1:.00f}/{self.total_train_batches:.00f}] {self.get_progress_bar_dict(trainer)}'
        
        self._update(con)
        
    def _update(self,con):
        print(con, end="\r", flush=True)
        
    def get_progress_bar_dict(self,trainer):
        tqdm_dict = trainer.progress_bar_dict
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
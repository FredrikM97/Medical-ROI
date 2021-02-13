from configs import load_config
from models import create_model
from datasets import create_dataset
from callbacks import ActivationMap, ConfusionMatrix
from progress import LitProgressBar

from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

import torch
from pytorch_model_summary import summary

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
        self.setup_trainer()
        
        
    def setup_trainer(self):
        self.trainer = pl.Trainer(
            max_epochs=self.config['model_params']['max_epochs'], 
            profiler=self.config['trainer_profiler'], 
            reload_dataloaders_every_epoch=self.config['trainer_reload_dataloaders_every_epoch'],
            gpus=self.get_gpu(), 
            logger=self.logger(),
            callbacks=self.callbacks(),
            progress_bar_refresh_rate=self.config['trainer_progress_bar_refresh_rate'],
            num_sanity_val_steps=self.config['trainer_num_sanity_val_steps'],
            benchmark=True,
            auto_lr_find=True,
            precision=self.config['trainer_precision']
            
        )
        
    def logger(self):
        return pl.loggers.TensorBoardLogger(self.config['logs']['tensorboard'], name=self.config['model_params']['model_name'] + "/" + self.config['model_params']['architecture'])
     
    def callbacks(self) -> list:
        lit_progress = LitProgressBar()
        activation_map = ActivationMap(self.model)
        checkpoint_callback = ModelCheckpoint(
            save_top_k=0, # disable?
            filename=self.config['logs']['checkpoint']+'{epoch}-{val_loss:.2f}',
            mode='min', 
        )
        return [lit_progress, checkpoint_callback, activation_map,ConfusionMatrix()]
    
    def get_gpu(self):
        return -1 if torch.cuda.is_available() else None
    
    def fit(self, cv=False) -> None:
        if self.config['agent']['kfold']:
            self.__fit_cv()
        else:
            self.__fit()
    
    def __fit(self):
        self.trainer.fit(
            self.model, 
            datamodule=self.dataset
        )
    
    def __fit_cv(self) -> None:
        print("Fitting with cv")
         
        while self.dataset._has_folds():
            # call fit
            fold_idx = self.dataset._get_fold()
            print(f"Validation on fold: {fold_idx}")
            self.setup_trainer()
            self.__fit()
            if self.trainer._state == TrainerState.INTERRUPTED: break

            # store metrics
            # get next fold
            self.dataset._next_fold()
     
    def model_summary(self):
        summary(self.model.model)
    def get_info():
        pass
        


    

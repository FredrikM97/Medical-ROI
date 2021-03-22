from src.utils import load
from . import models
from . import dataloader
from .callbacks import ActivationMapCallback, MetricCallback, CAMCallback, LitProgressBar
from src.utils import utils
from src.classifier.trainer import Trainer
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch

CONFIGDIR = 'conf/'

class Agent:
    def __init__(self, config_name:str, export:bool=True):
        print('Setup configurations...')
        
        self.load_config(config_name)
        self.load_logger()
        self.dataset = dataloader.create_dataset(**self.config['dataloader'])
        self.load_model()
        self.setup_trainer()
        
    def fit(self, cv=False) -> None:
        if self.config['model']['kfold']['enable']:
            self.__fit_cv()
        else:
            self.__fit()
            
    def setup_trainer(self):
        self.trainer = pl.Trainer(
            gpus=self.gpus, 
            logger=self.writer,
            callbacks=self.callbacks(),
            accelerator='ddp',
            **self.config['trainer']
        )
        
    def load_model(self):
        cfg_model = self.config['model']
        checkpoint_path = self.config['checkpoint_path']
        self.model = Trainer(checkpoint_path=checkpoint_path,**cfg_model)
        
    def save_model(self, filename=None):
        filename = filename if filename else 'checkpoint'
        self.trainer.save_checkpoint(filename+".ckpt")
    
    def load_config(self,config_name):
        configs = load.load_configs(CONFIGDIR)
        self.config = utils.merge_dict(configs['base']['classifier'],configs[config_name])

    def load_logger(self):
        self.writer = pl.loggers.TensorBoardLogger(
            self.config['logging']['tensorboard'], 
            name=self.config['model']['arch']['name'],
            default_hp_metric=False,
            log_graph=False,
        )
     
    def callbacks(self) -> list:
        lit_progress = LitProgressBar()
        #activation_map = ActivationMapCallback(self.model)
        checkpoint_callback = ModelCheckpoint()
        return [lit_progress, checkpoint_callback,MetricCallback()]
    
    @property
    def gpus(self):
        return -1 if torch.cuda.is_available() else None
    
    def __fit(self):
        self.trainer.fit(
            self.model, 
            datamodule=self.dataset
        )
    
    def __fit_cv(self) -> None:
        print("Fitting with cv")
        
        while self.dataset.kfold.has_folds():
            # call fit
            fold_idx = self.dataset.fold_idx
            print(f"Validation on fold: {fold_idx}")
            self.setup_trainer()
            self.__fit()
            if self.trainer._state == TrainerState.INTERRUPTED: break

            # store metrics
            # get next fold
            self.dataset._next_fold()
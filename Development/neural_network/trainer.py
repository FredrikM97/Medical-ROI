from configs import load_config
from .models import create_model
from .datasets import create_dataset
from .callbacks import ActivationMapCallback, MetricCallback, CAMCallback
from .utils.progress import LitProgressBar
from .utils import merge_dict

from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch

BASECONFIG = 'neural_network'

class Agent:
    def __init__(self, config_name:str, export:bool=True):
        print('Setup configurations...')
        self.config_name=config_name
        self.update_config()
        self.dataset = create_dataset(**self.config['dataset_params'])
        self.load_model()
        
    def fit(self, cv=False) -> None:
        if self.config['agent']['kfold']:
            self.__fit_cv()
        else:
            self.__fit()
            
    def _setup_trainer(self):
        self.trainer = pl.Trainer(
            max_epochs=self.config['model_params']['max_epochs'], 
            profiler=self.config['trainer_profiler'], 
            reload_dataloaders_every_epoch=self.config['trainer_reload_dataloaders_every_epoch'],
            checkpoint_callback=self.config['trainer_enable_checkpoints'],
            gpus=self.gpus, 
            logger=self.logger(),
            callbacks=self.callbacks(),
            progress_bar_refresh_rate=self.config['trainer_progress_bar_refresh_rate'],
            num_sanity_val_steps=self.config['trainer_num_sanity_val_steps'],
            accelerator='ddp',
            fast_dev_run=self.config['trainer_debug'],
            #precision=self.config['trainer_precision'],
        )
        
    def load_model(self):
        self.model = create_model(class_weights=self.dataset.weights, hp_metrics=self.config['logs']['hp_metrics'],**self.config['model_params'])
        self._setup_trainer()
        
    def save_model(self, filename=None):
        filename = filename if filename else 'checkpoint'
        self.trainer.save_checkpoint(filename+".ckpt")
    
    def update_config(self):
        self.config = merge_dict(load_config(BASECONFIG),load_config(self.config_name))
        
    def logger(self):
        print(self.config['logs']['tensorboard'])
        return pl.loggers.TensorBoardLogger(
            self.config['logs']['tensorboard'], name=self.config['model_params']['model_name'] + "/" + self.config['model_params']['architecture_name'], default_hp_metric=False,log_graph=False,
        )
     
    def callbacks(self) -> list:
        lit_progress = LitProgressBar()
        #activation_map = ActivationMapCallback(self.model)
        checkpoint_callback = ModelCheckpoint(
            #filename=self.config['logs']['checkpoint']+'{epoch}',
            #save_top_k=0, # disable?
        )
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

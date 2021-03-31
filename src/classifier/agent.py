from src.utils import load
from src.classifier.model import Model
from . import dataloader
from .callbacks import MetricCallback,DebugCallback,LitProgressBar
from src.utils.utils import merge_dict
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from src import BASEDIR
from typing import Tuple

def load_trainer(config_name:str, checkpoint_path:str=None):
    """ Load an trainer based on the configuration from config_name. If checkpoint_path is given it will overwride the checkpoint_path in the config file.
    If no checkpoint_path is given then the trainer will create a new model based on the config.
    
    Args:
        * config_name:str - Name of config. The existing config should end with .json to be valid for the trainer
        * checkpoint_path:str - Path to checkpoint. Home directory is within the src folder.
        
    Return:
        * Trainer object
        * Dataset object
        * Model object
    """
    model_config = load.load_config(config_name, dirpath=BASEDIR + "/conf/")[config_name]
    base_config = load.load_config('base', dirpath=BASEDIR + "/conf/")['base']['classifier']

    config = merge_dict(base_config,model_config)

    gpus_availible= 1 if torch.cuda.is_available() else None
    
    cfg_dataset = config['dataloader']
    cfg_model = config['model']
    
    # If we want to load a model directly without changing the config
    checkpoint_path = BASEDIR + checkpoint_path if checkpoint_path else BASEDIR + config['checkpoint_path']
    
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path} (checkpoint)..")
        model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path)
    else:
        model = Model(checkpoint_path=checkpoint_path,**cfg_model)
        
    dataset = dataloader.create_dataset(**cfg_dataset)
    
    
    logger = pl.loggers.TensorBoardLogger(
        BASEDIR +"/"+ config['logging']['tensorboard'], 
        name=config['model']['arch']['name'],
        default_hp_metric=False,
        log_graph=False,
    )
    
    
    callbacks = [
        LitProgressBar(),
        MetricCallback(),
        ModelCheckpoint(filename='checkpoint')
    ]
    
    trainer = pl.Trainer(
        gpus=gpus_availible, 
        logger=logger,
        callbacks=callbacks,
        accelerator='ddp',
        **config['trainer']
    )
    
    return trainer, dataset, model

def save_model(trainer, filename=None) -> None:
    """ Save the model from a trainer. If no filename it will default to "checkpoint"
    
    Args:
        * trainer:object - The trainer used to train a model
        * filename:str - The filename that the file should be saved as. The default directory is at the logger directory. 
    
    Return:
        * None
    """
    filename = filename if filename else 'checkpoint'
    trainer.save_checkpoint(filename+".ckpt")
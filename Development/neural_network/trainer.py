from src.utils import load
from src.classifier.trainer import Trainer
from . import dataloader
from .callbacks import MetricCallback,DebugCallback,LitProgressBar
from src.utils.utils import merge_dict

from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch

CONFIGDIR = 'conf/'

def load_trainer(config_name):
    configs = load.load_configs(CONFIGDIR)
    config = merge_dict(configs['base']['classifier'],configs[config_name])

    gpus_availible= 1 if torch.cuda.is_available() else None
    
    cfg_dataset = config['dataloader']
    cfg_model = config['model']
    checkpoint_path = config['checkpoint_path']
    
    dataset = dataloader.create_dataset(**cfg_dataset)
    model = Trainer(checkpoint_path=checkpoint_path,**cfg_model)
    
    logger = pl.loggers.TensorBoardLogger(
        config['logging']['tensorboard'], 
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

def save_model(trainer, filename=None):
    filename = filename if filename else 'checkpoint'
    trainer.save_checkpoint(filename+".ckpt")
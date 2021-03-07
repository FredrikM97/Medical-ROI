from configs import load_config
from .models import create_model
from .datasets import create_dataset
from .callbacks import MetricCallback,DebugCallback,LitProgressBar
from .utils import merge_dict

from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch

BASECONFIG = 'neural_network'

def load_trainer(config_name, load_model=False):
    config = merge_dict(load_config(BASECONFIG),load_config(config_name))

    gpus_availible= 1 if torch.cuda.is_available() else None

    cfg_dataset = config['dataset_params']
    cfg_model = config['model_params']
    
    dataset = create_dataset(**cfg_dataset)
    model = create_model(class_weights=dataset.weights, hp_metrics=config['logs']['hp_metrics'],**cfg_model)
    
    logger = pl.loggers.TensorBoardLogger(
        config['logs']['tensorboard'], 
        name=cfg_model['model_name'] + "/" +cfg_model['architecture_name'], 
        default_hp_metric=True,
        log_graph=False,
    )
    
    callbacks = [
        LitProgressBar(),
        MetricCallback()
    ]
    
    trainer = pl.Trainer(
            max_epochs=cfg_model['max_epochs'], 
            profiler=config['trainer_profiler'], 
            reload_dataloaders_every_epoch=config['trainer_reload_dataloaders_every_epoch'],
            checkpoint_callback=True,
            gpus=gpus_availible, 
            logger=logger,
            callbacks=callbacks,
            progress_bar_refresh_rate=0,
            num_sanity_val_steps=2,
            #accelerator='ddp',
            fast_dev_run=False,
            precision=config['trainer_precision'],
        )
    
    return trainer, dataset, model
import pytorch_lightning as pl
import torch
from src import BASEDIR
from typing import Tuple
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers
from src.classifier.dataloader.class_weight import ClassWeights,InitWeightDistribution
from src.utils.decorator import close_on_finish_decorator
from src.utils.utils import merge_dict
from src.utils import load
from src.classifier.model import Model

from .callbacks import MetricCallback,DebugCallback,LitProgressBar, CAMCallback
from . import dataloader
from datetime import datetime

class Agent:
    
    def __init__(self, config_name, base_config='base',checkpoint_path:str=None):
        self.checkpoint_path=checkpoint_path
        self._config = None
        self.base_config = base_config
        self.model = None
        self.dataloader = None
        
        
        self.creation_date = int(datetime.now().strftime('%Y%m%d%H%M%S'))

        self.load_config(config_name)
        self._weights_obj = ClassWeights(self._config['classes'], self._config['dataloader']['args']['delimiter'])
        
        self.load_dataloader()

    def load_config(self,config_name):
        """Init the config into object"""
        # Load all config settings
        model_config = load.load_config(config_name, dirpath=BASEDIR + "/conf/")
        base_config = load.load_config(self.base_config, dirpath=BASEDIR + "/conf/")
        config = merge_dict(base_config['classifier'],model_config)

        config.update({'classes':base_config['classes']})
         
        # Fix ROI bounding boxes and if dictionary then label them
        if config['model']['roi_hparams']['enable']:
            if isinstance(config['model']['roi_hparams']['boundary_boxes'], dict):
                config['model']['roi_hparams']['boundary_boxes'] = {config['classes'][key]:value for key,value in config['model']['roi_hparams']['boundary_boxes'].items()}
            elif isinstance(config['model']['roi_hparams']['boundary_boxes'], list):
                pass
            else:
                raise ValueError("boundary_boxes needs to be of type list or dict")
                
        self._config = config

    def load_model(self):
        """Init the model into object"""
        cfg_model = self._config['model']
    
        # If we want to load a model directly without changing the config
        checkpoint_path = self.checkpoint_path if self.checkpoint_path else cfg_model['checkpoint_path']

        # Check if dataset should be weighted
        
        if cfg_model['loss']['args']['weight']:
            self._weights_obj(self.dataloader.train_dataloader().dataset.labels)
        class_weights = self._weights_obj.weights

        # If checkpoint is enabled or if create a new model
        if checkpoint_path:
            #checkpoint_path = BASEDIR + checkpoint_path
            print(f"Loading model from {checkpoint_path} (checkpoint)..")
            model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path)
        else:
            model = Model(**cfg_model, class_weights=class_weights)

        # Set the init distribution for the weights
        if cfg_model['weight_distribution']:
            InitWeightDistribution(model)(cfg_model['weight_distribution'])
            
        self.model = model
    
    def load_dataloader(self):
        """Init the dataloader into object. If CV is enabled access the next folds with _dataloader.next_fold()"""
        cfg_dataset = self._config['dataloader']
    
        dataset = dataloader.create_dataset(classes=self._config['classes'], **cfg_dataset, seed=self._config['seed'])
        dataset.setup(stage='fit')
        self.dataloader = dataset
    
    
    
    def trainer(self, kfold_index:int=None):
        """Return a trainer object"""
        
        cfg_trainer = self._config['trainer']
    
        # Load logger
        logger = pl_loggers.TensorBoardLogger(
            self._config['logging']['tensorboard'], 
            name=f"{self._config['name']}/{self.creation_date}",
            default_hp_metric=False,
            log_graph=False,
        )

        # Setup callbacks
        callbacks = [
            LitProgressBar(),
            #MetricCallback(),
        ]
        callbacks.extend([getattr(pl_callbacks, key)(**values['args']) for key,values in cfg_trainer['callbacks'].items() if values['enable']])

        print("Enabled callbacks: ", [type(c).__name__ for c in callbacks])
        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else None, 
            logger=logger if cfg_trainer['tensorboard'] else None,
            callbacks=callbacks,
            **cfg_trainer["args"]
        )
        return trainer
    
    
    def run(self):
        self.load_model() 
        trainer = self.trainer(self.dataloader.kfold_index)
        print(f"Dataloader fold: {self.dataloader.kfold_index}")
        close_on_finish_decorator(trainer.fit, trainer.logger.log_dir, self.model, datamodule=self.dataloader, message=self._config)
        
        return trainer
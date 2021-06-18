"""
Agent class to load configs, model, dataloader and train the model.
"""

import pytorch_lightning as pl
import torch
from typing import Tuple
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers

import os
import contextlib
import itertools
import json


from datetime import datetime
from threading import Lock
from src.utils.decorator import HiddenPrints
from src.cam import CAM
from src.files.file import create_directory
from . import callbacks as lc_callbacks
from . import dataloader
from src import BASEDIR
from src.dataloader.class_weight import ClassWeights,InitWeightDistribution
from src.utils.decorator import close_on_finish_decorator
from src.types.dictionary import merge_dict
from src.files import load
from src.model import Model
from src.files.file import write



class Agent:
    """ """
    
    def __init__(self, config_name:str=None, base_config:str='base',checkpoint_path:str=None, print_enabled:bool=False):
        """Init function for Agent

        Parameters
        ----------
        config_name : str
            (Default value = None)
        base_config : str
            (Default value = 'base')
        checkpoint_path : str
            (Default value = None)
        print_enabled : bool
            (Default value = False)

        Returns
        -------

        
        """
        self._config = None
        self.print_enabled = print_enabled
        self.base_config = base_config
        self.model = None
        self.dataloader = None
        self.trainer = None
        
        self.creation_date = int(datetime.now().strftime('%Y%m%d%H%M%S'))

        self.load_config(config_name, checkpoint_path=checkpoint_path)
        self._weights_obj = ClassWeights(self._config['classes'])
        
        # Dummy loading of everything to initiate all variables
        self.load_dataloader()
   

    def load_config(self,config_name:str, checkpoint_path:str=None) -> None:
        """Init the config into object

        Args:
          config_name(str): 
          checkpoint_path(str, optional): (Default value = None)

        Returns:

        Raises:

        """
        # Load all config settings
        if config_name == None and checkpoint_path == None:
            raise ValueError("Both config_name and checkpoint_path cant be None!")
        
        elif config_name == None and checkpoint_path != None:
            model_config = {}
        else:  
            model_config = load.load_config(config_name, dirpath=BASEDIR + "/conf/")
        
        base_config = load.load_config(self.base_config, dirpath=BASEDIR + "/conf/")
        config = merge_dict(base_config['classifier'],model_config)

        config.update({'classes':base_config['classes']})
         
        # Overwrite if no checkpoint_path is selected outside of the config
        config['checkpoint_path'] = checkpoint_path if checkpoint_path else config['checkpoint_path']
        
        # Fix ROI bounding boxes and if dictionary then label them
        if config['model']['roi_hparams']['enable']:
            if isinstance(config['model']['roi_hparams']['boundary_boxes'], dict):
                config['model']['roi_hparams']['boundary_boxes'] = {config['classes'][key]:value for key,value in config['model']['roi_hparams']['boundary_boxes'].items()}
            elif isinstance(config['model']['roi_hparams']['boundary_boxes'], list):
                pass
            else:
                raise ValueError("boundary_boxes needs to be of type list or dict")
        
        pl.utilities.seed.seed_everything(seed=config["seed"], workers=True)
        
        self._config = config

    def load_model(self):
        """Init the model into object"""
        cfg_model = self._config['model']

        # If checkpoint is enabled or if create a new model
  
        if self._config['checkpoint_path']:
            model = Model.load_from_checkpoint(checkpoint_path=self._config['checkpoint_path'])
        else:
            # Check if dataset should be weighted
            if cfg_model['loss']['args']['weight']:
                self._weights_obj(self.dataloader.train_dataloader().dataset.labels)
            class_weights = self._weights_obj.weights
            
            model = Model(**cfg_model, class_weights=class_weights)

            # Set the init distribution for the weights
            if cfg_model['weight_distribution']:
                InitWeightDistribution(model,cfg_model['weight_distribution'])
                
            if self.print_enabled: print("Architecture [{0}] was created".format(type(model).__name__))
        
        self.model = model
    
    def load_dataloader(self) -> None:
        """Init the dataloader into object. If CV is enabled access the next folds with _dataloader.next_fold()

        Args:

        Returns:

        Raises:

        """
        cfg_dataset = self._config['dataloader']
    
        dataset = dataloader.create_dataset(classes=self._config['classes'], **cfg_dataset, seed=self._config['seed'])
        dataset.setup() #stage='fit'
        self.dataloader = dataset
    
    
    
    def load_trainer(self) -> None:
        """Load a trainer to class

        Args:

        Returns:

        Raises:

        """
        
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
            lc_callbacks.LitProgressBar()
        ]
        callbacks.extend([getattr(pl_callbacks, key)(**values['args']) for key,values in cfg_trainer['callbacks'].items() if values['enable']])

        if self.print_enabled: print("Enabled callbacks: ", [type(c).__name__ for c in callbacks])
        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else None, 
            logger=logger if cfg_trainer['tensorboard'] else None,
            callbacks=callbacks,
            **cfg_trainer["args"]
        )
        self.trainer = trainer
    
    
    def run(self):
        """Load the model and trainer, and run the agent"""
        self.load_model() 
        self.load_trainer()
        self.print_enabled: print(f"Dataloader fold: {self.dataloader.kfold_index}")
        
        
        close_on_finish_decorator(self.trainer.fit, self.trainer.logger.log_dir, self.model, datamodule=self.dataloader, message=self._config)        
        return self.trainer
    
    def print_info(self):
        """ """
        print(
                f"Checkpoint: {self._config['checkpoint_path']}\n"
                f"Seed: {self._config['seed']}\n"
                f"{self.model}\n\n{self.dataloader}"
                f"{self.model.roi_model if self.model.roi_model != None else ''}" 
        )

    def save_hparams(self):
        """ """
        dir_path = f"{self.trainer.logger.log_dir.rsplit('/',1)[0]}/"
        create_directory(dir_path) 
        with open(f"{dir_path}hparams.json", "w") as write_file:
            json.dump(self._config, write_file, indent = 6)

        
    

class ThreadSafeReloadedModel:
    """ """
    
    def __init__(self, checkpoint_path, cam_type, cam_kwargs={}):
        """

        Parameters
        ----------
        checkpoint_path :
            
        cam_type :
            
        cam_kwargs :
            (Default value = {})

        Returns
        -------

        
        """
        self.checkpoint_path = checkpoint_path
        self.cam_type = cam_type
        self.cam_kwargs = cam_kwargs
        self.trainer = Agent(checkpoint_path=checkpoint_path)
        self.lock = Lock()
        
    def __call__(self):
        """ """
        self.lock.acquire()
        self.trainer.load_model()
        model = self.trainer.model
        self.lock.release()
        return CAM(model,cam_type=self.cam_type, cam_kwargs=self.cam_kwargs) 
        
    
    def get_dataloader(self):
        """Access the dataloder object"""
        return self.trainer.dataloader
    
    def get_validation_images(self, observe_classes:list=None):
        """

        Args:
          observe_classes(list, optional): (Default value = None)

        Returns:

        Raises:

        """
        fileset = self.get_dataloader().val_dataloader().dataset
        if observe_classes != None: 
            return ([idx, image, patient_class, target_class] for (idx, (image, patient_class)), target_class in itertools.product(enumerate(fileset),observe_classes)) 
        return fileset


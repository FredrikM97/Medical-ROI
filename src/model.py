"""
Model class for the agent and training of different architectures.
"""

#from ..architectures import create_architecture #testModel
from . import models

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics as pl_metrics

import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


from src.files import preprocess
from src.display.plot import confusion_matrix

from src import BASEDIR

class Model(pl.LightningModule): 
    """ """
    def __init__(self,class_weights:torch.Tensor=None,hp_metrics:list=None,loss={}, roi_hparams={"enable":False,'input_shape':None, 'bounding_boxes':[]},**hparams):
        """

        Parameters
        ----------
        class_weights : torch.Tensor
            (Default value = None)
        hp_metrics : list
            (Default value = None)
        loss :
            (Default value = {})
        roi_hparams :
            (Default value = {"enable":False,'input_shape':None, 'bounding_boxes':[]})
        **hparams :
            

        Returns
        -------

        
        """
        super().__init__() 
        self.save_hyperparameters()
        self.model = models.create_model(**self.hparams['arch'])
        self.roi_enabled = roi_hparams['enable']
        self.hp_metrics = hp_metrics
        self.roi_model = None
        self.criteria = nn.__dict__[loss['type']](weight=class_weights)
        
        
        if self.roi_enabled:
            # Dont import unless we want to use RoiTransform.. (Compability without cuda 11.1)
            from src.utils.transforms import RoiTransform
            self.roi_model = RoiTransform(**roi_hparams)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.valid_metrics = self.metric_object('/val')
            self.test_metrics = self.metric_object('/test')

    def metric_object(self, postfix:bool=None, compute_on_step:bool=False) -> pl_metrics.MetricCollection:
        return pl_metrics.MetricCollection({
            "Accuracy":pl_metrics.Accuracy(average='micro', compute_on_step=compute_on_step),
            "Precision":pl_metrics.Precision(num_classes=3, average='micro',compute_on_step=compute_on_step),
            "Recall":pl_metrics.Recall(num_classes=3, average='micro',compute_on_step=compute_on_step),
            "AUROC":pl_metrics.AUROC(num_classes=3, average='macro',compute_on_step=compute_on_step),
            "ConfusionMatrix":pl_metrics.ConfusionMatrix(num_classes=3, normalize='true',compute_on_step=compute_on_step),
            "Specificity":pl_metrics.Specificity(num_classes=3, average='micro',compute_on_step=compute_on_step),
            "AUROC/class":pl_metrics.AUROC(num_classes=3, average=None,compute_on_step=compute_on_step)
        }, postfix=postfix)
    
    def on_train_start(self):
        """ """
        if self.logger:
            self.logger.log_hyperparams(self.hparams, {metric:0 for metric in self.hp_metrics})

    def forward(self, x):
        """

        Parameters
        ----------
        x :
            

        Returns
        -------

        
        """
        return self.model(x)
    
    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """

        Parameters
        ----------
        batch : dict
            
        batch_idx : int
            

        Returns
        -------

        
        """
        x, target = batch
        # Since we might want to apply ROI at any time we can enable and disable it here. ROIAlign needs to be on the GPU!
        if self.roi_enabled:
            x, target = self.roi_model(x, target)

        logits = self.forward(x)
        loss = self.criteria(logits,target)
        
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """

        Parameters
        ----------
        batch : dict
            
        batch_idx : int
            

        Returns
        -------

        
        """
        x, target = batch
        # Since we might want to apply ROI at any time we can enable and disable it here. ROIAlign needs to be on the GPU!
        if self.roi_enabled:
            x, target = self.roi_model(x, target)

        
        logits = self.forward(x)
        
        loss = self.criteria(logits,target)
        preds = F.softmax(logits,dim=1)
        self.valid_metrics(preds,target)
        self.log('loss/val',loss)
  
        return {'loss/val':loss}
    
    def test_step(self, batch: dict, batch_idx: int) -> dict:
        """

        Parameters
        ----------
        batch : dict
            
        batch_idx : int
            

        Returns
        -------

        
        """
        x, target = batch
        # Since we might want to apply ROI at any time we can enable and disable it here. ROIAlign needs to be on the GPU!
        if self.roi_enabled:
            x, target = self.roi_model(x, target)

        logits = self.forward(x)
        
        loss = self.criteria(logits,target)
        preds = F.softmax(logits,dim=1)
        self.test_metrics(preds,target)
        self.log('loss/test',loss)
  
        return {'loss/test':loss}

    def training_epoch_end(self, outputs):
        """

        Parameters
        ----------
        outputs :
            

        Returns
        -------

        
        """
        self.log('loss/train',torch.stack([x['loss'] for x in outputs]).mean())
        
    def validation_epoch_end(self,outputs):
        """

        Parameters
        ----------
        outputs :
            

        Returns
        -------

        
        """
        
        metrics = self.valid_metrics.compute()
        auc = metrics.pop("AUROC/class/val")
        
        self.logger.experiment.add_figure(f"confmat/val", confusion_matrix(metrics.pop('ConfusionMatrix/val')),self.current_epoch)
        
        self.log_dict({f'AUROC/{key}/val/':val for key,val in zip(['CN','MCI','AD'],auc)})

        self.log_dict(metrics)
        self.valid_metrics.reset()

    def test_epoch_end(self, outputs):
        metrics = self.test_metrics.compute()
        auc = metrics.pop("AUROC/class/test")
        
        self.logger.experiment.add_figure(f"confmat/test", confusion_matrix(metrics.pop('ConfusionMatrix/test')),self.current_epoch)
        
        self.log_dict({f'AUROC/{key}/test/':val for key,val in zip(['CN','MCI','AD'],auc)})
        self.log_dict(metrics)
        self.test_metrics.reset()

        
    def configure_optimizers(self):
        """ """
        # Note: dont use list if only one item.. Causes silent crashes
        optim = torch.optim.__dict__[self.hparams.optimizer['type']]
        optimizer = optim(self.model.parameters(), **self.hparams.optimizer['args'])
        
        return {
            'optimizer': optimizer,
        }

    def __str__(self):
        """ """
        return (f"""{"Architecture: [{0}]".format(type(self.model).__name__)}\n"""
                f"""***Defined hyperparameters:***\n{self.hparams}""")

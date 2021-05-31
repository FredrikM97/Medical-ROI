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


from src.utils import preprocess
from src.utils.plot import confusion_matrix

from src import BASEDIR
from src.classifier.metric import MetricTracker

def create_model(checkpoint_path=None,**cfg_model):
    if checkpoint_path:
        assert os.path.isfile(checkpoint_path), "The provided checkpoint_path is not valid! Does it exist?"
        print(f"Loading model from {checkpoint_path} (checkpoint)..")
        return trainer.load_from_checkpoint(checkpoint_path=checkpoint_path)
    else:
        return models.create_model(**cfg_model['arch'])

class Model(pl.LightningModule): 
    def __init__(self,class_weights:torch.Tensor=None,hp_metrics:list=None,loss={}, roi_hparams={"enable":False,'roi_shape':None, 'bounding_boxes':[]},**hparams):
        super().__init__() 
        self.save_hyperparameters()
        self.model = create_model(**self.hparams)
        self.roi_enabled = roi_hparams['enable']
        self.hp_metrics = hp_metrics

        self.criteria = nn.__dict__[loss['type']](weight=class_weights)
        if self.roi_enabled:
            # Dont import unless we want to use RoiTransform.. (Compability without cuda 11.1)
            from src.utils.transforms import RoiTransform
            self.roi_model = RoiTransform(**roi_hparams)

        model_metrics = pl_metrics.MetricCollection([
            pl_metrics.Accuracy(average='micro', compute_on_step=False),
            pl_metrics.Precision(num_classes=3, average='micro',compute_on_step=False),
            pl_metrics.Recall(num_classes=3, average='micro',compute_on_step=False),
            pl_metrics.AUROC(num_classes=3, average='macro',compute_on_step=False),
            pl_metrics.ConfusionMatrix(num_classes=3, normalize='true',compute_on_step=False),
            pl_metrics.Specificity(num_classes=3, average='micro',compute_on_step=False),
        ])
        
        self.valid_metrics = model_metrics.clone()
        self.valid_dummy_metric = pl_metrics.AUROC(num_classes=3, average=None,compute_on_step=False)
        #MetricTracker()
        
        print(f"***Defined hyperparameters:***\n{self.hparams}")
        
    def on_train_start(self):
        if self.logger:
            self.logger.log_hyperparams(self.hparams, {metric:0 for metric in self.hp_metrics})
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        # Since we might want to apply ROI at any time we can enable and disable it here. ROIAlign needs to be on the GPU!
        if self.roi_enabled:
            x, target = self.roi_model(x, target)

        logits = self.forward(x)
        loss = self.criteria(logits,target)
        
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        # Since we might want to apply ROI at any time we can enable and disable it here. ROIAlign needs to be on the GPU!
        if self.roi_enabled:
            x, target = self.roi_model(x, target)

        
        logits = self.forward(x)
        
        loss = self.criteria(logits,target)
        preds = F.softmax(logits,dim=1)
        
        self.valid_metrics(preds, target)
        self.valid_dummy_metric(preds,target)
        self.log('loss/val',loss)
  
        return {'loss/val':loss}
  
    def training_epoch_end(self, outputs):
        self.log('loss/train',torch.stack([x['loss'] for x in outputs]).mean())
        
    def validation_epoch_end(self,outputs):
        
        metrics = self.valid_metrics.compute()

        self.logger.experiment.add_figure(f"confmat/val", confusion_matrix(metrics.pop('ConfusionMatrix')),self.current_epoch)
        
        self.logger.experiment.add_scalars('Class_AUROC/val', {key:val for key,val in zip(['CN','MCI','AD'],self.valid_dummy_metric.compute())},self.global_step)
        self.log_dict({str(key)+'/val': val for key, val in metrics.items()}, on_step=False, on_epoch=True)
        
        self.valid_metrics.reset()
        self.valid_dummy_metric.reset()
        
    def configure_optimizers(self):
        # Note: dont use list if only one item.. Causes silent crashes
        optim = torch.optim.__dict__[self.hparams.optimizer['type']]
        optimizer = optim(self.model.parameters(), **self.hparams.optimizer['args'])
        
        #return optimizer
        return {
            'optimizer': optimizer,
        }
            
    #def loss_fn(self,out,target):
    #    #return nn.CrossEntropyLoss(weight=self.loss_class_weights)(out,target)
    #    return self.criteria(out,target)
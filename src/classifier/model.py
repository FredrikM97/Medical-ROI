#from ..architectures import create_architecture #testModel
from . import models

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import preprocess
from src.utils.plot import ROC

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
        #self.loss_class_weights = class_weights if loss_weight_balance else None
        #class_weights=None,loss_weight_balance=None, part of input
        if self.roi_enabled:
            # Dont import unless we want to use RoiTransform.. (Compability without cuda 11.1)
            from src.segmentation import RoiTransform
            self.roi_model = RoiTransform(**roi_hparams)
        self.metricsTracker = MetricTracker()
        
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
        loss = self.loss_fn(logits, target) 
        
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        # Since we might want to apply ROI at any time we can enable and disable it here. ROIAlign needs to be on the GPU!
        if self.roi_enabled:
            x, target = self.roi_model(x, target)

        
        logits = self.forward(x)
        loss = self.loss_fn(logits, target) 

        _, predicted = torch.max(logits, 1)
        probability = F.softmax(logits,dim=0)
        
        self.log('loss/val_step',loss)
        
        self.metricsTracker(
            predicted,
            target,
            probability,
            loss
        )
        
        return {'loss/val':loss, 'predicted/val':predicted, 'target/val':target, "probability/val":probability}
  
    def training_epoch_end(self, outputs):
        self.log('loss/train',torch.stack([x['loss'] for x in outputs]).mean())
        
    def validation_epoch_end(self,outputs):
        pred, target, prob, loss = self.metricsTracker.compute()
    
        cm_plot(
            self, 
            pl.metrics.functional.confusion_matrix(pred, target, num_classes=3), 
            prefix='val'
        )
        
        (auc, fpr, tpr), roc_fig = ROC(pl.metrics.functional.roc(prob, target, num_classes=3))
        '''
        self.logger.log_metrics(
            {
                #f"auc/val": auc,
                f"specificity/val":1-fpr,
                f"sensitivity/val":tpr
            },step=self.current_epoch
        ) 
        '''
        
        self.log("specificity/val",1-fpr)
        self.log("sensitivity/val",tpr)
        self.log('auc/val',auc)
        
        self.log('loss/val',loss.mean())
        self.log('accuracy/val',pl.metrics.functional.accuracy(pred, target))
        
        #self.logger.log_metrics({'loss/val':loss.mean(),'accuracy/val':pl.metrics.functional.accuracy(pred, target)}, self.current_epoch)
        self.logger.experiment.add_figure(f"ROC/val", roc_fig, self.current_epoch)
        
        # Reset the states.. we dont want to keep it in memory! As of pytorch-lightning 1.2 this is not done automatically!
        self.metricsTracker.reset()

    def configure_optimizers(self):
        # Note: dont use list if only one item.. Causes silent crashes
        optim = torch.optim.__dict__[self.hparams.optimizer['type']]
        optimizer = optim(self.model.parameters(), **self.hparams.optimizer['args'])
        
        #return optimizer
        return {
            'optimizer': optimizer,
        }
            
    def loss_fn(self,out,target):
        #return nn.CrossEntropyLoss(weight=self.loss_class_weights)(out,target)
        return self.criteria(out,target)
    

def cm_plot(trainer, cm, prefix=''):
        fig = plt.figure(figsize=(20,20))
        ax = sns.heatmap(preprocess.tensor2numpy(cm), annot=True, annot_kws={"size": 12})
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        trainer.logger.experiment.add_figure(f"confmat/{prefix}", fig,trainer.current_epoch)
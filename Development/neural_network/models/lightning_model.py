from architectures import create_architecture #testModel

import pytorch_lightning as pl
import torch.nn as nn
import torch
from pytorch_lightning.metrics import functional as FM
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics.classification import ConfusionMatrix
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
# https://pytorch-lightning.readthedocs.io/en/latest/metrics.html
class LightningModel(pl.LightningModule): 
    def __init__(self, **hparams):
        super().__init__() 
        self.save_hyperparameters()
        
        self.model = create_architecture(architecture=self.hparams.architecture,input_channels=1, num_classes=3)
        
        self.save_hyperparameters()
        
        self.loss_class_weights = self.hparams.class_weights if self.hparams.loss_weight_balance else None
        
        # These have internal memory
        self.train_metric = Accuracy(compute_on_step=False)
        self.val_metric = Accuracy(compute_on_step=False)
        self.train_cm = ConfusionMatrix(3, compute_on_step=False)
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        pred = self.forward(x)
  
        loss = self.loss_fn(pred, target) 
        
        self.train_metric(pred,target)
        self.train_cm(pred,target)
        return {'loss':loss, 'predicted':pred, 'target':target}
  
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, target) 
        
        self.val_metric(pred,target)
        return {'loss':loss, 'predicted':pred, 'target':target}
    
    def training_epoch_end(self, outputs):
        # logging histograms
        self.custom_histogram_adder()
        
        loss = avg_metric(outputs, 'loss')
        acc = self.train_metric.compute()
        #self._log_cm()
        self._log_scalar('train', loss, acc)
    
    def validation_epoch_end(self, outputs):
        loss = avg_metric(outputs, 'loss')
        acc = self.val_metric.compute()
        
        self._log_scalar('val', loss, acc)
        
    def configure_optimizers(self):
        # Note: dont use list if only one item.. Causes silent crashes
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.opt_weight_decay, amsgrad=self.hparams.opt_amsgrad)
        return optimizer
    
    def loss_fn(self,out,target):
        return nn.CrossEntropyLoss(weight=self.loss_class_weights)(out,target)

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.model.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)
    
    def _log_scalar(self, prefix, loss, acc):
        self.logger.experiment.add_scalar(f"loss/"+prefix,
                                            loss,
                                            self.current_epoch)
        
        self.logger.experiment.add_scalar(f"accuracy/"+prefix,
                                            acc,
                                            self.current_epoch)

def avg_metric(outputs, metric):
    return torch.stack([x[metric] for x in outputs]).mean()


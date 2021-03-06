#from ..architectures import create_architecture #testModel

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class Trainer(pl.LightningModule): 
    def __init__(self, model=None,class_weights=None,loss_weight_balance=None,hp_metrics:list=None,**hparams):
        super().__init__() 
        self.save_hyperparameters()
        #self.example_input_array = torch.rand(64, 1, 79, 69, 79)
        self.model = architecture

        self.hp_metrics = hp_metrics
        self.save_hyperparameters()
        
        self.loss_class_weights = class_weights if loss_weight_balance else None
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {metric:0 for metric in self.hp_metrics})
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, target) 
        
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, target) 

        _, predicted = torch.max(logits, 1)
        probability = F.softmax(logits,dim=0)
        
        return {'loss/val':loss, 'predicted/val':predicted, 'target/val':target, "probability/val":probability}
    
    def training_epoch_end(self, outputs):
        #Just here to fix a bug..
        pass
    
    def configure_optimizers(self):
        # Note: dont use list if only one item.. Causes silent crashes
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.opt_weight_decay, amsgrad=self.hparams.opt_amsgrad)
        return optimizer
    
    def loss_fn(self,out,target):
        return nn.CrossEntropyLoss(weight=self.loss_class_weights)(out,target)
    
    
    
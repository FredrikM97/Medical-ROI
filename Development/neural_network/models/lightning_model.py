from architectures import create_architecture #testModel

import pytorch_lightning as pl
import torch.nn as nn
import torch
from pytorch_lightning.metrics import functional as FM
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy
#from pytorch_lightning.metrics.classification import ConfusionMatrix
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

class LightningModel(pl.LightningModule): 
    def __init__(self, **hparams):
        super().__init__() 
        self.save_hyperparameters()
        
        self.model = create_architecture(architecture=self.hparams.architecture,input_channels=1, num_classes=3)
        
        self.save_hyperparameters()
        
        self.loss_class_weights = self.hparams.class_weights if self.hparams.loss_weight_balance else None
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, target) 
        
        return {'loss':loss}
  
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, target) 

        _, predicted = torch.max(logits, 1)
        probability = F.softmax(logits,dim=0)
        
        return {'loss':loss, 'predicted':predicted, 'target':target, "probability":probability}

    def configure_optimizers(self):
        # Note: dont use list if only one item.. Causes silent crashes
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.opt_weight_decay, amsgrad=self.hparams.opt_amsgrad)
        return optimizer
    
    def loss_fn(self,out,target):
        return nn.CrossEntropyLoss(weight=self.loss_class_weights)(out,target)
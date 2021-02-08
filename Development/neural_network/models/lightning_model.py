from architectures import testModel

import pytorch_lightning as pl
import torch.nn as nn
import torch
from pytorch_lightning.metrics import functional as FM

# https://pytorch-lightning.readthedocs.io/en/latest/metrics.html
class LightningModel(testModel): 
    def __init__(self, hparams):
        
        if hparams is None:
            hparams = {}
        super().__init__(input_channels=1, num_classes=3) #hparams
        
        self.loss = nn.CrossEntropyLoss() 
        self.lr = hparams.get('lr')
        self.save_hyperparameters()
        
        self.train_metrics = torch.nn.ModuleDict({
            'accuracy': pl.metrics.Accuracy(),
        })
        self.val_metrics = torch.nn.ModuleDict({
            'accuracy': pl.metrics.Accuracy(),
        })
        self.test_metrics = torch.nn.ModuleDict({
            'accuracy': pl.metrics.Accuracy()
        })
        
        
    def training_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        pred = self.forward(x) 
        loss = self.loss(pred, target) 
        
        # log values
        #self.logger.experiment.add_scalar('Train/Loss', loss)  
        
        #Calculate metrics
        logs = {f"{'train'}/{key}":metric(pred,target) for key,metric in self.train_metrics.items()}
        logs.update({'train/loss': loss})
        self.log_dict(logs)
        
        return {'loss': loss}
  
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        pred = self.forward(x) 
        loss = self.loss(pred, target) 
        
        logs = {f"{'val'}/{key}":metric(pred,target) for key,metric in self.val_metrics.items()}
        logs.update({'val/loss': loss})
        self.log_dict(logs, prog_bar=True)
  
    def test_step(self, batch: dict, batch_idx: int) -> dict:
        logs = {f"{'test'}/{key}":metric(pred,target) for key,metric in self.test_metrics.items()}
        
        self.log_dict(logs)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return [optimizer]

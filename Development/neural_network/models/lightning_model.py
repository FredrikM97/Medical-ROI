from architectures import testModel

import pytorch_lightning as pl
import torch.nn as nn
import torch
from pytorch_lightning.metrics import functional as FM

# https://pytorch-lightning.readthedocs.io/en/latest/metrics.html
class LightningModel(pl.LightningModule): 
    def __init__(self, hparams):
        super().__init__() 
        
        self.model = testModel(input_channels=1, num_classes=3)
        self.loss = nn.CrossEntropyLoss() 
        self.lr = hparams.get('lr')
           
        self.save_hyperparameters()
        
    def training_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        pred = self.model(x) 
        
        # log values
        #self.logger.experiment.add_scalar('Train/Loss', loss)  
        metrics = self.step_metrics(pred, target)
        self.log_dict({f'train/{k}':v for k,v in metrics})
        return metrics
  
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        pred = self.model(x) 
        
        metrics = self.step_metrics(pred, target)
        self.log_dict({f'val/{k}':v for k,v in metrics})
        
        return metrics
    
    def test_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        pred = self.model(x) 

        metrics = self.step_metrics(pred, target)
        self.log_dict({f'test/{k}':v for k,v in metrics})
        
        return metrics
    """
    def training_epoch_end(self, outputs):
        self.log_epoch_end('train', outputs)
        
    def validation_epoch_end(self, outputs):
        self.log_epoch_end('val', outputs)
        
    def test_epoch_end(self, outputs):
        self.log_epoch_end('test', outputs)
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return [optimizer]
    
    def step_metrics(self, pred, target):
        loss = self.loss(pred, target) 
        accuracy = FM.accuracy(pred, target)
        
        return {'loss':loss, 'accuracy':accuracy}
    
    def log_epoch_end(self,prefix, outputs):
        loss = avg_metric(outputs, 'loss')
        acc = avg_metric(outputs, 'accuracy')

        self.log_dict({f'{prefix}/loss':loss, f'{prefix}/accuracy':acc})

def avg_metric(outputs, metric):
    return torch.stack([x[metric] for x in outputs]).mean()


import pytorch_lightning as pl
import torch

class storeMetrics(pl.metrics.Metric):
    def __init__(self,compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=False)
        self.add_state('predicted',default=[], dist_reduce_fx=None)
        self.add_state('target',default=[], dist_reduce_fx=None)
        self.add_state('probability',default=[], dist_reduce_fx=None)
        self.add_state('loss',default=torch.tensor(0.), dist_reduce_fx=None)
        self.add_state('num_batch',default=torch.tensor(0.), dist_reduce_fx=None)
        
    def update(self, predicted, target, probability, loss):
        self.predicted.append(predicted)
        self.target.append(target)
        self.probability.append(probability)
        self.loss+=loss.mean()
        
        self.num_batch +=1
        
    def compute(self):
        # Returns: predicted, target, loss, probability, epoch
        return (
            torch.cat(self.predicted, dim=0),
            torch.cat(self.target, dim=0),
            torch.cat(self.probability, dim=0),
            self.loss/self.num_batch
        )

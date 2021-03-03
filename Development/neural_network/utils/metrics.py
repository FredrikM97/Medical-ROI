import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import to_cpu_numpy

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

class meanMetric(pl.metrics.Metric):
    def __init__(self,compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=False)
        self.add_state('val',default=torch.tensor(0.), dist_reduce_fx='mean')
        self.add_state('num',default=torch.tensor(0.), dist_reduce_fx='mean')
    def update(self, loss):
        self.val+=loss.mean()
        self.num +=1
        
    def compute(self):
        return self.val/self.num
    
def ROC(roc_classes, prefix=''):
    # Returns (auc, fpr, tpr), roc_fig
    fig = plt.figure(figsize = (10,7))
    lw = 2
    colors = np.array(['aqua', 'darkorange', 'cornflowerblue'])
    fpr, tpr, threshold = roc_classes
    
    metric_list = np.zeros(3)
    for i in range(len(roc_classes)):
        auc = to_cpu_numpy(pl.metrics.functional.auc(fpr[i],tpr[i]))
        
        _fpr = to_cpu_numpy(fpr[i])
        _tpr = to_cpu_numpy(tpr[i])
        
        metric_list[0]+=auc
        metric_list[1]+=_fpr.mean()
        metric_list[2]+=_tpr.mean()
        
        plt.plot(_fpr, _tpr, color=colors[i], lw=lw,
                 label='ROC curve of class {0} (area={1:0.2f} tpr={2:0.2f} fpr={3:0.2f})'
                 ''.format(i,auc, _tpr.mean(), 1-_fpr.mean())) #
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.close()
    
    metric_list = metric_list/3
    
    return metric_list, fig
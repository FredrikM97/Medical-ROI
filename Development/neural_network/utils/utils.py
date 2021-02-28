import torch
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np
def label_encoder(labels):
    "Convert list of string labels to tensors"
    from sklearn import preprocessing
    import torch
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
                  
    return torch.as_tensor(targets)

def get_availible_files(path, contains:str=''):
    return [f for f in os.listdir(path) if contains in f]

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
    
def to_cpu_numpy(data):
    # Send to CPU. If computational graph is connected then detach it as well.
    return data.detach().cpu().numpy() if data.requires_grad else data.cpu().numpy()
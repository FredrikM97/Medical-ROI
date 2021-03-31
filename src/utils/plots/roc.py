from src.utils import preprocess
from . import plot
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

@plot.figure_decorator  
def ROC(roc_classes, prefix='', fig=None):
    # Returns (auc, fpr, tpr), roc_fig
    #fig = plt.figure(figsize = (10,7))
    lw = 2
    colors = np.array(['aqua', 'darkorange', 'cornflowerblue'])
    fpr, tpr, threshold = roc_classes
    
    metric_list = np.zeros(3)
    for i in range(len(roc_classes)):
        auc = preprocess.tensor2numpy(pl.metrics.functional.auc(fpr[i],tpr[i]))
        
        _fpr = preprocess.tensor2numpy(fpr[i])
        _tpr = preprocess.tensor2numpy(tpr[i])
        
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
    
    metric_list = metric_list/3
    
    return metric_list, fig
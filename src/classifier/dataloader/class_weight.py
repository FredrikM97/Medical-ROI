"""
Classes related to the evaluation of weights for data and distribution of the model.
"""

from src.utils import preprocess
import torch
import sklearn
import torch.nn as nn

class ClassWeights:
    def __init__(self, classes:dict):
        self.classes=classes
        self.weights = None
        
    def __call__(self, labels):

        # Compute weights
        self.weights = torch.from_numpy(sklearn.utils.class_weight.compute_class_weight('balanced', classes=list(self.classes.values()), y=labels)).float().cuda()
    
    def numpy(self):
        return self.weights.numpy()
    
class InitWeightDistribution:
    
    #def __init__(self, model):
    #    self.model = model
    def __new__(self, model, dist):
        getattr(self, dist)(self,model)
        
    
    def uniform(self,m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)
        
    def normal(self,model):
        for m in model.model.modules():
            #classname = m.__class__.__name__
            # for every Linear layer in a model..
            #print("Children of normal:", type(m))
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
      
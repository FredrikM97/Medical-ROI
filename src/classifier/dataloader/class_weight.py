from src.utils import preprocess
import torch
import sklearn
import torch.nn as nn

class ClassWeights:
    def __init__(self, classes:dict, delimiter:str):
        self.classes=classes
        self.delimiter=delimiter
        self.weights = None
        
    def calculate(self, images):
        self.labels = preprocess.filename2labels(images, self.classes, self.delimiter)#get_labels(images, self._classes, self._delimiter)

        # Compute weights
        self.weights = torch.from_numpy(sklearn.utils.class_weight.compute_class_weight('balanced', classes=list(self.classes.values()), y=self.labels)).float().cuda()
    
    def numpy(self):
        return self.weights.numpy()
    
class InitWeightDistribution:
    
    def __init__(self, model):
        self.model = model
        
    def __call__(self, dist):
        return getattr(self, dist)(self.model)
        
        
    def uniform(self,m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)
        
    def normal(self,model):
        for m in model.modules():
            #classname = m.__class__.__name__
            # for every Linear layer in a model..
            print("Children of normal:", m)
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.constant_(m.bias, 0)

            """
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
                
            """
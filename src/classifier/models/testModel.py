import pytorch_lightning as pl
import torch.nn as nn
from collections import OrderedDict

class testModel(nn.Module):
    """ """
    def __init__(self,num_channels:int=1,num_classes:int=3, **kwargs):
        """

        Parameters
        ----------
        num_channels : int
            (Default value = 1)
        num_classes : int
            (Default value = 3)
        **kwargs :
            

        Returns
        -------

        
        """
        # Model for input (batch, channel, slices, with, height) -> (batch,1,79, 96, 79)
        super().__init__()
        print(num_channels)
        self.conv_layer1 = self._conv_layer_set(num_channels, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(64*18*22*18, 64)
        self.fc2 = nn.Linear(64, num_classes)
        #self.fc3 = nn.Linear(128, num_classes)
        self.batch=nn.BatchNorm3d(64)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        """

        Parameters
        ----------
        in_c :
            
        out_c :
            

        Returns
        -------

        
        """
        conv_layer = nn.Sequential(nn.ModuleDict([
            ('conv',nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0)),
            ('leakyrelu',nn.LeakyReLU()),
            ('maxpool',nn.MaxPool3d((2, 2, 2))),
        ]))
        return conv_layer
    

    def forward(self, x):
        """

        Parameters
        ----------
        x :
            

        Returns
        -------

        
        """
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.batch(out)
        out1 = out.view(-1, self.num_flat_features(out))
        #out1 = out.view(out.size(0), 128*18*22*18)
        out = self.fc1(out1)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out

    def num_flat_features(self, x):
        """

        Parameters
        ----------
        x :
            

        Returns
        -------

        
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
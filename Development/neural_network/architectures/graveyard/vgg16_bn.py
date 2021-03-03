from architectures.blocks import conv_bn, Conv3dAuto,activation_func
import torch.nn as nn
from functools import partial

__all__ = ['VGG16BN',]

class VGG16BN(nn.Module):
    def __init__(self,input_channels=1, num_classes=3, **kwargs):
        super().__init__()  

        self.encoder = nn.Sequential(
            nn.Sequential(
            *self._conv_layer(input_channels,64),
            *self._conv_layer(64,64),
            ),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(64),
            nn.Sequential(
                *self._conv_layer(64,128),
                *self._conv_layer(128,128),  
            ),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(128),
            nn.Sequential(
                *self._conv_layer(128,256),
                *self._conv_layer(256,256),
                *self._conv_layer(256,256),
            ),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(256),
            nn.Sequential(
                *self._conv_layer(256,512),
                *self._conv_layer(512,512),
                *self._conv_layer(512,512),
            ),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(512),
            nn.Sequential(
                *self._conv_layer(512,512),
                *self._conv_layer(512,512),
                *self._conv_layer(512,512), 
            ),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(512),
        )
        

        self.dense = nn.Sequential(
            nn.Linear(512*2*2*2, 4096), # Depends on input size
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        ) 
        
        
    def forward(self, x):
        #x = self.block1(x)
        x = self.encoder(x)
        x = x.view(-1, self.num_flat_features(x))
        
        x = self.dense(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def _conv_layer(self, input, output, kernel=3):
        return nn.ModuleList([
            Conv3dAuto(input, output, kernel),
            nn.LeakyReLU(),
        ])

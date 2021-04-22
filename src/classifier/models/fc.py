import torch
import torch.nn as nn


class testModel(nn.Module):
    def __init__(self,num_channels:int=1,num_classes:int=3, image_shape=(40,40,40),**kwargs):
        # Model for input (batch, channel, slices, with, height) -> (batch,1,79, 96, 79)
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(image_shape[0]*image_shape[1]*image_shape[2], 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)
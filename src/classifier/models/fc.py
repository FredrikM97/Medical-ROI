import torch
import torch.nn as nn


class fc(nn.Module):
    def __init__(self,num_channels:int=1,num_classes:int=3, input_shape=(3,3,3),**kwargs):
        # Model for input (batch, channel, slices, with, height) -> (batch,1,79, 96, 79)
        super().__init__()
        self.input_shape = input_shape
        self.classifier = nn.Sequential(
            nn.Linear(input_shape[0]*input_shape[1]*input_shape[2], 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        print("FC shape",x.shape)
        x = torch.flatten(x, 1)
        print("FC shape after",x.shape)
        #print("asdads",x.shape,self.input_shape)
        return self.classifier(x)
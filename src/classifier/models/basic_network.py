import torch
import torch.nn as nn


class basic(nn.Module):
    def __init__(self,num_channels:int=1,num_classes:int=3, input_shape=(79,95,79),**kwargs):
        # Model for input (batch, channel, slices, with, height) -> (batch,1,79, 96, 79)
        super().__init__()
        self.input_shape = input_shape
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2),
            nn.Conv3d(16, 16, kernel_size=3, stride=1),
            
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1),
            nn.Conv3d(32, 64, kernel_size=3, stride=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 64, kernel_size=3,stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((5, 5, 5))
        self.fc = nn.Sequential(
            nn.Linear(8000, 512),
            #nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Linear(256, 3),
            #nn.ReLU(True),
        )

    def forward(self, x):
        #print(x.shape)
        x = self.cnn(x)
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return self.fc(x)
import pytorch_lightning as pl
import torch.nn as nn

class testModel(pl.LightningModule):
    def __init__(self, input_channels=1, num_classes=3):
        # Model for input (batch, channel, slices, with, height) -> (batch,1,79, 96, 79)
        super().__init__()
        self.conv_layer1 = self._conv_layer_set(input_channels, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(64*18*22*18, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out1 = out.view(out.size(0), 64*18*22*18)
        out = self.fc1(out1)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out

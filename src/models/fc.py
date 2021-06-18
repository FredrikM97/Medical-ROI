import torch
import torch.nn as nn


class fc(nn.Module):
    """ """
    def __init__(self,num_channels:int=1,num_classes:int=3, input_shape=(16,14,11),**kwargs):
        """

        Parameters
        ----------
        num_channels : int
            (Default value = 1)
        num_classes : int
            (Default value = 3)
        input_shape :
            (Default value = (16,14,11))
        **kwargs :
            

        Returns
        -------

        
        """
        # Model for input (batch, channel, slices, with, height) -> (batch,1,79, 96, 79)
        super().__init__()
        self.input_shape = input_shape
        self.classifier = nn.Sequential(
            nn.Linear(input_shape[0]*input_shape[1]*input_shape[2], 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """

        Args:
          x: 

        Returns:

        Raises:

        """
        x = torch.flatten(x, 1)
        return self.classifier(x)
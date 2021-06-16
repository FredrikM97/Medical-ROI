from torch import nn
import torch.nn.functional as F

class customCNN(nn.Module):
    """ """
    
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs :
            

        Returns
        -------

        
        """
        super().__init__()
        
        self.conv_1 = nn.Conv3d(1, 16, kernel_size = (3, 3, 3))
        self.pool_1 = nn.MaxPool3d((2, 2, 2))
        self.batch_1 = nn.BatchNorm3d(16)
        
        self.conv_2 = nn.Conv3d(16, 32, kernel_size = (3, 3, 3))
        self.pool_2 = nn.MaxPool3d((2, 2, 2))
        self.batch_2 = nn.BatchNorm3d(32)
        
        self.conv_3 = nn.Conv3d(32, 64, kernel_size = (3, 3, 3))
        self.pool_3 = nn.MaxPool3d((2, 2, 2))
        self.batch_3 = nn.BatchNorm3d(64)
        
        self.fc_1 = nn.Linear(40960, 128) #35840
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 3)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """

        Parameters
        ----------
        x :
            

        Returns
        -------

        
        """
        
        #print(x.shape)
        
        x = self.conv_1(x)
        x = F.leaky_relu(x)
        x = self.pool_1(x)
        x = self.batch_1(x)
        
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        x = self.pool_2(x)
        x = self.batch_2(x)
        
        x = self.conv_3(x)
        x = F.leaky_relu(x)
        x = self.pool_3(x)
        x = self.batch_3(x)
        
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc_2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        return self.fc_3(x)
    
    def num_flat_features(self, x):
        """

        Parameters
        ----------
        x :
            

        Returns
        -------

        
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
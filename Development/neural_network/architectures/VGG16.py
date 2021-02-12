import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self,input_channels=1, num_classes=3, **kwargs):
        super().__init__()
        self.convDropout = nn.Dropout3d(p=0.4)  
        
        self.conv1 = nn.Conv3d(input_channels, 64, 3, padding = 1)
        self.conv2 = nn.Conv3d(64, 64, 3, padding = 1)
        
        self.conv3 = nn.Conv3d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv3d(128, 128, 3, padding = 1)
        
        self.conv5 = nn.Conv3d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv3d(256, 256, 3, padding = 1)
        self.conv7 = nn.Conv3d(256, 256, 3, padding = 1)
        
        self.conv8 = nn.Conv3d(256, 512, 3, padding = 1)
        self.conv9 = nn.Conv3d(512, 512, 3, padding = 1)
        self.conv10 = nn.Conv3d(512, 512, 3, padding = 1)
        
        self.conv11 = nn.Conv3d(512, 512, 3, padding = 1)
        self.conv12 = nn.Conv3d(512, 512, 3, padding = 1)
        self.conv13 = nn.Conv3d(512, 512, 3, padding = 1)
        
        self.fc1 = nn.Linear(512*2*2*2, 4096) # Depends on input size
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.convDropout(x)
        x = F.max_pool3d(x, 2)
        #BN?
        
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.convDropout(x)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = self.convDropout(x)
        x = F.max_pool3d(x, 2)
        #BN?
        
        x = self.conv5(x)
        x = F.leaky_relu(x)
        x = self.convDropout(x)
        x = self.conv6(x)
        x = F.leaky_relu(x)
        x = self.convDropout(x)
        x = self.conv7(x)
        x = F.leaky_relu(x)
        x = self.convDropout(x)
        x = F.max_pool3d(x, 2)
        #BN?
        
        x = self.conv8(x)
        x = F.leaky_relu(x)
        x = self.conv9(x)
        x = F.leaky_relu(x)
        x = self.conv10(x)
        x = F.leaky_relu(x)
        x = F.max_pool3d(x, 2)
        #BN?
        
        x = self.conv11(x)
        x = F.leaky_relu(x)
        x = self.conv12(x)
        x = F.leaky_relu(x)
        x = self.conv13(x)
        x = F.leaky_relu(x)
        x = F.max_pool3d(x, 2)
        #BN?
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.5)
      
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.5)
        
        x = self.fc3(x)
        x = F.softmax(x, 1)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
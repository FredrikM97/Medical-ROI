import torch
#from .config import NNConfig
import os

def setup():
    # Move tensor to the GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #conf = NNConfig()
    return device, conf
        
def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
        
def save_model():
    pass

def load_model():
    pass

def get_nii_files(srcdir):
    return [
            path + '/' + filename
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ] 
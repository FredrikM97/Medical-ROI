import torch

def setup():
    # Move tensor to the GPU if available
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
        
def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
        
def save_model():
    pass

def load_model():
    pass
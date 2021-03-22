from . import utils
from torchcam import cams
import torch
    
__all__ = ['get_cam']

    
def get_cam(model, image, extractor_name='SmoothGradCAMpp', input_shape=(1,79,224,224),target_layer=None, observed_class=None):
    assert tuple(image.shape) == input_shape[1:], f"Got image shape: {image.shape} expected: {input_shape[1:]}"
    assert model.device == image.device, f"Model and image are not on same device: Model: {model.device} Image: {image.device}"
    extractor = {
        'CAM':cams.CAM,
        'ScoreCAM':cams.ScoreCAM,
        'SSCAM':cams.SSCAM,
        'ISCAM':cams.ISCAM,
        'GradCAM':cams.GradCAM,
        'GradCAMpp':cams.GradCAMpp,
        'SmoothGradCAMpp':cams.SmoothGradCAMpp,
        'Saliency':Saliency
    }[extractor_name]
    
    # Not sure If this should be done?
    for param in model.parameters():
        param.requires_grad = True
        
    model = model.eval()
    img_tensor = utils.batchisize_to_5D(image)
    img_tensor.requires_grad = True

    # Hook your model before the forward pass
    
    cam_extractor = extractor(model,input_shape=input_shape, target_layer=target_layer)
    
    # By default the last conv layer will be selected
    #with torch.no_grad():
    score = model(img_tensor)
    
    # If we want to overwride the label we want to observe!
    predicted_label = score.squeeze(0).argmax().item() if not observed_class else observed_class
    
    # Retrieve the CAM
    if not extractor_name == 'Saliency':
        activation_map = cam_extractor(predicted_label, score)
    else:
        activation_map = cam_extractor(predicted_label, score, img_tensor)
    
    return utils.to_cpu_numpy(activation_map), predicted_label

class Saliency:
    def __init__(self, model, target_layer=None,input_shape=(1,79,224,224)):
        self.model = model
        self.target_layer = None
        self.input_shape = input_shape
        
    def __call__(self,label, preds, image):       
        #assert image.shape == input_shape
        for param in self.model.parameters():
            param.requires_grad = False

        
        #preds = model(image)
        score, indices = torch.max(preds, 1)

        score.backward()

        # Get max along channel axis
        slc, _ = torch.max(torch.abs(image.grad[0]), dim=0)

        # normalize to [0..1]
        activation_map = utils.normalize(slc)

        # Not sure If this should be done?
        for param in self.model.parameters():
            param.requires_grad = True

        return activation_map
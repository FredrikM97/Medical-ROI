from ..utils import utils
from torchcam import cams
import torch
"""
def generate_CAM(model, image, backend='gcampp', label=0, data_shape=(79,224,224)):

    # Inject model with M3d-CAM
    model = medcam.inject(model, backend=backend,output_dir="logs/attention_maps", label=label,layer='model.layer4',save_maps=True,data_shape=data_shape)

    model.eval()
    
    output = model(image)
    
    mask = medcam.medcam_utils.normalize(model.medcam_dict['current_attention_map'])
    mask = resize(mask.squeeze(0).squeeze(0),data_shape)
    return mask
"""
def get_cam(model, image, extractor_name='SmoothGradCAMpp', img_shape=(1,79,224,224)):
    assert tuple(image.shape) == img_shape[1:], f"Got image shape: {image.shape} expected: {img_shape[1:]}"
    assert model.device == image.device, f"Model and image are not on same device: Model: {model.device} Image: {image.device}"
    extractor = {
        'CAM':cams.CAM,
        'ScoreCAM':cams.ScoreCAM,
        'SSCAM':cams.SSCAM,
        'ISCAM':cams.ISCAM,
        'gradcam':cams.gradcam,
        'GradCAM':cams.GradCAM,
        'GradCAMpp':cams.GradCAMpp,
        'SmoothGradCAMpp':cams.SmoothGradCAMpp,
    }[extractor_name]
    
    # Not sure If this should be done?
    for param in model.parameters():
        param.requires_grad = True
        
    model = model.eval()
    img_tensor = utils.batchisize_to_5D(image)
    img_tensor.requires_grad = True

    # Hook your model before the forward pass
    cam_extractor = extractor(model,input_shape=img_shape)
    
    # By default the last conv layer will be selected
    #with torch.no_grad():
    score = model(img_tensor)
    # Retrieve the CAM
    activation_map = cam_extractor(score.squeeze(0).argmax().item(), score)
    return utils.to_cpu_numpy(activation_map)

def saliency(model, image, img_shape=(79,224,224)):
    assert image.shape == img_shape, f"Wrong image shape, Expected: {img_shape}"
    assert model.device == image.device, f"Model and image are not on same device: Model: {model.device} Image: {image.device}"
   
    for param in model.parameters():
        param.requires_grad = False
    
    image = utils.batchisize_to_5D(image)
    
    model.eval()

    image.requires_grad = True
    
    assert image.shape == img_shape
    preds = model(image)
    score, indices = torch.max(preds, 1)

    score.backward()
    
    # Get max along channel axis
    slc, _ = torch.max(torch.abs(image.grad[0]), dim=0)
    
    # normalize to [0..1]
    activation_map = utils.normalize(slc)

    # Not sure If this should be done?
    for param in model.parameters():
        param.requires_grad = True
        
    return utils.to_cpu_numpy(activation_map)
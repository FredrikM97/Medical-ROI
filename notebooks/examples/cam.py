from src.classifier.utils.cam import get_cam
from src.classifier.utils.cam_plots import plot_CAM_grid
from src.classifier.utils import move_to_device, tensor2numpy
from src.classifier import trainer

def example_plot_cam()
    plot_grid(example_cam())

def example_cam():
    extractor_name:str='SmoothGradCAMpp'
    img_size=(79,224,224),
    cmap='jet'
    alpha=0.3
    target_label='CN'
    target_layer = 'model.layer4'
    agent = trainer.Agent()
    
    label_to_class = {
        0:'CN',
        1:'MCI',
        2:'AD'
    }
    class_to_label = {v: k for k, v in label_to_class.items()}
    observed_class = observed_class if observed_class else load_image
    
    # Load and Resize image
    image = sample_images()[target_label]
    image = torch.from_numpy(resize(image, img_size)).float()
    
    # Get model from agent
    model = agent.model
    
    # Move to cuda device
    model, image = move_to_device(model, image, 'cuda')
    
    mask, predicted_label = get_cam(
        model, image, extractor_name=extractor_name, target_layer=target_layer, observed_class=class_to_label[observed_class]
    )
    
    return image, mask, target_layer, predicted_label, target_label, extractor_name


def plot_grid(image, mask, target_layer, predicted_label, target_label, extractor_name):
    fig = plot_CAM_grid(
            tensor2numpy(image), 
            mask,
            layer=target_layer, 
            label=label_to_class[predicted_label], 
            observed_class=target_label,
            extractor=extractor_name
        )

def sample_images():
    return = {
        'CN':nib.load('data/SPM_categorised/AIH/CN/CN_ADNI_998.nii').get_fdata,
        'MCI':nib.load('data/SPM_categorised/AIH/MCI/MCI_ADNI_1586.nii').get_fdata,
        'AD':nib.load('data/SPM_categorised/AIH/AD/AD_ADNI_2975.nii').get_fdata
    }
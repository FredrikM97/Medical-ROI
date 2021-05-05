from src.utils import utils, preprocess

from torchcam import cams
import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Union, List
import enum
from src.utils.cmap import parula_map
import torchvision
import matplotlib.pyplot as plt


from src.utils import plot


class SaliencyMap:
    def __init__(self, model, target_layer=None,input_shape=(1,79,95,79)):
        self.model = model
        self.target_layer = None
        self.input_shape = input_shape
        
    def __call__(self, class_idx,class_scores): 
        raise NotImplemented
        # Return an activation map which is not normalized
        #assert image.shape == input_shape
        #for param in self.model.parameters():
        #    param.requires_grad = False
        
        #class_idx = class_scores.argmax()
        #output_max = class_scores[0, class_idx]

        # Do backpropagation to get the derivative of the output based on the image
        #output_max.backward()

        #preds = model(image)
        #score, indices = torch.max(preds, 1)
        #score, indices = torch.max(class_scores, 1)
        #score.retain_grad()
        #score.backward()

        # Get max along channel axis
        #activation_map, _ = torch.max(torch.abs(score.grad[0]), dim=0)
        #print(activation_map)
        #activations, _ = torch.max(score.grad.data.abs(), dim=1) 
        #print(self.model.grad.data.abs())
        #return activation_map

class CAM_TYPES(enum.Enum):
    CAM = cams.CAM
    ScoreCAM = cams.ScoreCAM
    SSCAM = cams.SSCAM
    ISSCAM = cams.ISCAM
    GradCAM = cams.GradCAM
    GradCAMpp = cams.GradCAMpp
    SmoothGradCAMpp = cams.SmoothGradCAMpp
    #Saliency = SaliencyMap
    
class CAM:
    """
    Example usage:
    trainer, dataset, model = load_trainer('resnet50')
    tmp_image = nib.load('../data/SPM_categorised/AIH/AD/AD_ADNI_2975.nii').get_fdata()
    tmp = cam.CAM(CAMS.GradCAM.value, model, tmp_image)
    
    tmp.plot(tmp.class_scores, [0,1,2], class_label="AD")
    """
    def __init__(self, cam:str, model, input_shape:Tuple[int,int,int,int]=(79,95,79), target_layer:str=None, device:str='cuda') -> None:
        """ Init object for CAM extraction. 
        
        This code only works in the colormap is grayscale so C=1
        The dataloaded support the axial view and return it as default
        
        Args:
            cam (str): CAM_TYPES reference
            model (nn.Module): Which model that CAM should be extracted from
            input_shape (Tuple[D,H,W]): Expected shape of the input image. Color and batch should not be included.
            target_layer (str), optional: Which layer that should be selected.
            device (str): cuda or cpu.
        """
        super().__init__()
        #self.cam = cam
        self.model = model
        self.extractor = cam(model, input_shape=(1,*input_shape), target_layer=target_layer)
        self.input_shape = input_shape
        self.device = device

    def evaluate(self,input_image:np.ndarray) -> Tuple[Tensor, int]:
        """ "Calculate the class scores and the highest probability of the target class
            
        Args:
        
        Return:
            * Tuple containing all the probabilities and the best probability class
        """
 
        # Apply preprocessing
        image = preprocess.preprocess_image(input_image)
        image = preprocess.batchisize_to_5D(image)
        image_tensor = torch.from_numpy(image).float()
        
        self.model.to(self.device)
        image_tensor = image_tensor.to(self.device)
        # Check that image have the correct shape
        assert tuple(image_tensor.shape) == (1, 1, *self.input_shape), f"Got image shape: {image_tensor.shape} expected: {(1, 1, *self.input_shape)}"
        assert self.model.device == image_tensor.device, f"Model and image are not on same device: Model: {self.model.device} Image: {image_tensor.device}"
  
        self.model.eval()
        class_scores = self.model(image_tensor)
        return class_scores, class_scores.squeeze(0).argmax().item()
    
    def activation_map(self, class_idx:int=None, class_scores:Tensor=None) -> np.ndarray:
        """ Retrieve the map based on the score from the model
        
        Return:
            * Tensor with activations from image with shape Tensor[D,H,W]
        """

        return preprocess.tensor2numpy(self.extractor(class_idx, class_scores, normalized=False))
       
    def grid_class(self, class_scores:Tensor, class_idx:Union[List[int],int], input_image:np.ndarray, max_num_slices:int=16,pad_value=0.5, normalized=True, nrow=10,**kwargs) -> Tuple[Tensor, Tensor]:
        """Creates a grid based on a class_idx."""
        
        image_process = lambda _image: preprocess.preprocess_image(_image, normalized=normalized)
        if isinstance(class_idx, list) and len(class_idx) == 1:
            class_idx = class_idx[0]

        if isinstance(class_idx, list):
            grid_img = torch.hstack([
                preprocess.to_grid(image_process(input_image), max_num_slices=max_num_slices,pad_value=pad_value,nrow=nrow)
            ]*len(class_idx))
            
            grid_mask = torch.hstack([
                preprocess.to_grid(image_process(self.activation_map(idx, class_scores)), max_num_slices=max_num_slices,pad_value=pad_value,nrow=nrow)
                for idx in class_idx
            ])
        
            
        elif isinstance(class_idx, int):
            grid_mask = preprocess.to_grid(image_process(self.activation_map(class_idx, class_scores)), max_num_slices=max_num_slices,pad_value=pad_value,nrow=nrow)
            grid_img = preprocess.to_grid(image_process(input_image), max_num_slices=max_num_slices,pad_value=pad_value,nrow=nrow)
            
        else:
            raise ValueError(f"Expected class_idx of type list or int, Got: {type(class_idx)}")
            
        return grid_img, grid_mask
        
    def plot(self, class_scores:Tensor, class_idx:Union[List[int],int], input_image:np.ndarray,cmap=parula_map, alpha=0.7, class_label:str=None, predicted_override=None, max_num_slices:int=16,nrow=10):
        """Create a plot from the given class activation map and input image. CAM is calculated from the models weights and the probability distribution of each class.
        
        Args:
            class_scores (Tensor): Tensor of probability for each class 
            class_idx (Union[List[int],int]): Index of highest class_score probability 
            input_image (np.ndarray): image to the evaluated
            cmap (Color object), optional: The cmap to use in plot 
            alpha (int), optional: Alpha value for opacity 
            class_label (str), optional: Which class the image belongs to.
            predicted_override (bool), optional: If class_idx should be overwritten and plot each class instead
            max_num_slices (int), optional: Number of total slices that should be plotted. Combine multiple slices if image slices are more than number of max_num_slices.
            
        Return:
            output (Figure): Figure reference to plot
        """
        class_idx = class_idx if isinstance(class_idx, list) else [class_idx]
        
        def default_settings(axis, predicted_label):
            classes = {
                0:'CN',
                1:'MCI',
                2:'AD'
            }
            title_list = [out for out, con in [
                (f'{type(self.model.model).__name__}',True),
                (f'{type(self.extractor).__name__}',True),
                (f'Patient: {class_label}',class_label),
                (f'Predicted: {classes[predicted_label]}',predicted_label),
                (f'Overrided',predicted_override)] if con != None
            ]
            axis.set_title(', '.join(title_list))
            
            axis.set_axis_off()
            axis.set_xticklabels([])
            axis.set_yticklabels([])
                
        fig, axes = plt.subplots(ncols=len(class_idx),nrows=1,figsize=(len(class_idx)*8,8))
        fig.subplots_adjust(hspace=0)
        
        
        if len(class_idx) == 1:
            image, mask = self.grid_class(class_scores, class_idx[0],input_image,max_num_slices=max_num_slices,nrow=nrow)
            axes.imshow(image,cmap='Greys_r', vmin=image.min(), vmax=image.max())
            im = axes.imshow(mask,cmap=cmap, alpha=alpha,vmin=mask.min(), vmax=mask.max()) 
            default_settings(axes,class_idx[0])
        else:
            for i, idx in enumerate(class_idx):
                image, mask = self.grid_class(class_scores, idx,input_image, max_num_slices=max_num_slices,nrow=nrow)
                axes[i].imshow(image,cmap='Greys_r', vmin=image.min(), vmax=image.max())
                im = axes[i].imshow(mask,cmap=cmap, alpha=alpha,vmin=mask.min(), vmax=mask.max()) 
                default_settings(axes[i], idx)
                
            
        
        # Remove axis data to show colorbar more clean
        ax = axes.ravel().tolist() if len(class_idx) > 1 else axes
        plt.subplots_adjust(wspace=0.01, hspace=0)
        cbar = fig.colorbar(im, ax=ax, shrink=1)

        
        return fig
    
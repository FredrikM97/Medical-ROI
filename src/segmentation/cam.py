from src.utils import utils, preprocess
from src.utils.plots import plot
from torchcam import cams
import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Union, List
import enum
from src.utils.plots.parula import parula_map
import torchvision
import matplotlib.pyplot as plt

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

class CAMS(enum.Enum):
    CAM = cams.CAM
    ScoreCAM = cams.ScoreCAM
    SSCAM = cams.SSCAM
    ISSCAM = cams.ISCAM
    GradCAM = cams.GradCAM
    GradCAMpp = cams.GradCAMpp
    SmoothGradCAMpp = cams.SmoothGradCAMpp
    #Saliency = SaliencyMap
    
class CAM(plot.Plot):
    """
    Example usage:
    trainer, dataset, model = load_trainer('resnet50')
    tmp_image = nib.load('../data/SPM_categorised/AIH/AD/AD_ADNI_2975.nii').get_fdata()
    tmp = cam.CAM(CAMS.GradCAM.value, model, tmp_image)
    
    tmp.plot(tmp.class_scores, [0,1,2], class_label="AD")
    """
    def __init__(self, cam:str, model, image:np.ndarray, input_shape:Tuple[int,int,int,int]=(1,79,95,79), target_layer:str=None) -> None:
        super().__init__()
        self.cam = cam
        self.model = model
        self.image = image
        self.extractor = cam(model, input_shape=input_shape, target_layer=target_layer)
        self.input_shape = input_shape
        self.class_scores, self.class_idx = self.evaluate()

    def _cam_scores(self) -> Tensor:
        """ Sign it over so we dont have to call self each time. Note: Is this a copy? 
            
            Return a score vector with the propbability of each class
        """
        
        # Apply preprocessing
        image = self.preprocess(self.image)
        image = preprocess.batchisize_to_5D(image)
        image_tensor = torch.from_numpy(image).float()
        
        image_tensor.to(self.model.device)
        # Check that image have the correct shape
        assert tuple(image_tensor.shape) == (1, *self.input_shape), f"Got image shape: {image_tensor.shape} expected: {(1, *self.input_shape)}"
        #assert self.model.device == image.device, f"Model and image are not on same device: Model: {self.model.device} Image: {image.device}"
        
        #image_tensor.requires_grad = True
        
        # Enable grad and configure model
        #for param in self.model.parameters():
        #    param.requires_grad = False
        
        self.model.eval()
        class_scores = self.model(image_tensor)
        return class_scores
    
    def _class_idx(self, class_score:Tensor):
        """Convert score vector with probabilities into the maximum score value"""
        return class_score.squeeze(0).argmax().item()
    
    def evaluate(self) -> Tuple[Tensor, int]:
        """Calculate the class scores and the highest probability of the target class
        
        Return:
            * Tuple containing all the probabilities and the best probability class
        """
        class_scores = self._cam_scores() 
        return class_scores, self._class_idx(class_scores)
    
    def activation_map(self, class_idx:int=None, class_scores:Tensor=None) -> np.ndarray:
        """ Retrieve the map based on the score from the model
            Return a Tensor with activations from image
        """
        
        return utils.tensor2numpy(self.extractor(class_idx, class_scores))
       
    def grid_class(self, class_scores:Tensor, class_idx:Union[List[int],int], max_num_slices:int=16) -> Tuple[Tensor, Tensor]:
        """Creates a grid based on a class_idx."""
        if isinstance(class_idx, list) and len(class_idx) == 1:
            class_idx = class_idx[0]

        
        if isinstance(class_idx, list):
            grid_img = torch.hstack([
                self.grid(self.image, max_num_slices=max_num_slices)
                for _ in class_idx
            ])
            
            print(class_scores, class_idx)
            grid_mask = torch.hstack([
                self.grid(self.activation_map(class_idx, class_scores), max_num_slices=max_num_slices)
                for class_idx in class_idx
            ])
        
            
        elif isinstance(class_idx, int):
            grid_mask = self.grid(self.activation_map(class_idx, class_scores), max_num_slices=max_num_slices)
            grid_img = self.grid(self.image, max_num_slices=max_num_slices)
            
        else:
            raise ValueError(f"Expected class_idx of type list or int, Got: {type(class_idx)}")
            
        return grid_img, grid_mask
        
    def plot(self, class_scores:Tensor, class_idx:Union[List[int],int],cmap=parula_map, alpha=0.3, class_label:str=None, predicted_override=None, max_num_slices:int=16):
        """Create a plot from the given class activation map and input image. CAM is calculated from the models weights and the probability distribution of each class."""
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
            
        fig, axes = plt.subplots(ncols=len(class_idx),nrows=1,figsize=(len(class_idx)*8,8))
        fig.subplots_adjust(hspace=0)
        
        if len(class_idx) == 1:
            image, mask = self.grid_class(class_scores, class_idx[0])
            axes.imshow(image,cmap='Greys_r')
            im = axes.imshow(mask,cmap=cmap, alpha=alpha) 
            default_settings(axes,class_idx[0])
        else:
            for i, idx in enumerate(class_idx):
                image, mask = self.grid_class(class_scores, idx)
                axes[i].imshow(image,cmap='Greys_r')
                im = axes[i].imshow(mask,cmap=cmap, alpha=alpha) 
                default_settings(axes[i], idx)
        
        # Remove axis data to show colorbar more clean
        for ax in axes.flat:
            ax.set_axis_off()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    
        plt.subplots_adjust(wspace=0.01, hspace=0)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1)

        
        return fig
    
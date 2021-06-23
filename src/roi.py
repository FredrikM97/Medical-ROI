"""
Transformation of ROI with the help of RoiAlign.
"""


from typing import List, Tuple, Union

import numpy as np
import torch
from roi_align import RoIAlign
from torchvision.ops._utils import convert_boxes_to_roi_format

from src.files.preprocess import tensor2numpy


class RoiTransform:
    """Apply ROI transform to shange shape of images and Transform boundary boxes to correct format."""
    
    def __init__(self, output_shape:Tuple[int,int,int]=None, boundary_boxes:Union[List[Tuple[int,int,int,int,int,int]]]=None, batch_size=6,**args):
        """ Init RoiTransform object.
        
        Parameters
        ----------
        output_shape : Tuple[int,int,int]
             (Default value = None)
        boundary_boxes : Union[List[Tuple[int,int,int,int,int,int]]]
             (Default value = None)
        batch_size :
             (Default value = 6)
        **args :


        Returns
        -------

        """
        
        if not boundary_boxes: raise ValueError("bounding_boxes list can not be empty!")
        
        self.batch_size = batch_size
        self.roi = RoIAlign(output_shape,spatial_scale=1.0,sampling_ratio=-1)
        

        if isinstance(boundary_boxes, list):
            self.boundary_boxes = convert_boxes_to_roi_format([torch.stack([torch.Tensor(x) for x in boundary_boxes])])
            self.num_bbox = len(boundary_boxes)
        elif isinstance(boundary_boxes, dict):
            self.boundary_boxes = {key:convert_boxes_to_roi_format([torch.stack([torch.Tensor(x) for x in value])]) for key,value in boundary_boxes.items()}
            
            self.num_bbox = sum(map(len, self.boundary_boxes.values()))
        else:
            raise ValueError("boundary_boxes needs to be of type list or dict")
            
    def __call__(self, x:'tensor', y) -> 'Tuple[tensor, tensor]':
        """Expect to take an y of integer type and if boundary_boxes are a dict then the key should be a numeric value.

        Parameters
        ----------
        x : torch.Tensor
            Input value. Expect shape (B,C,D,H,W)
        y : Tensor
            Target value

        Returns
        -------

        
        """

        if isinstance(self.boundary_boxes, list):
            image_rois = self.roi.forward(x,torch.cat(x.shape[0]*[self.boundary_boxes.to(x.device)]))#.detach()
            y = self.num_bbox*y
        elif isinstance(self.boundary_boxes, dict):
            image_rois = self.roi.forward(x,torch.cat([self.boundary_boxes[one_target].to(x.device) for one_target in tensor2numpy(y)]))

            y = torch.from_numpy(np.concatenate([len(self.boundary_boxes[one_target])*[one_target] for one_target in tensor2numpy(y)])).to(x.device)

        else:
            raise ValueError("boundary_boxes needs to be of type list or dict")

        return image_rois, y
    
    def __str__(self) -> str:
        """ """
        return (
            f"\n\n***Defined ROI-Transformer:***\n"
            f"Number of BBoxes: {self.num_bbox}\n"
            f"BBox Count: {len(self.boundary_boxes) if isinstance(self.boundary_boxes, list) else ', '.join([f'{x}:{y}' for x,y in zip(self.boundary_boxes.keys(),map(len, self.boundary_boxes.values()))])}"
        )
    
def roi_align(image, boxes:list, output_shape:Tuple=(40,40,40), displayed:bool=False) -> 'torch.Tensor':
    """Create aligned image rois for the neural network

    Args:
      image: Image of shape Tuple[D,H,W]
      boxes(list): List of features (z0,y0,z1,y1,x0,x1). Shape is expected based on the input of ROIAlign
      output_shape(Tuple, optional): (Default value = (40,40,40))
      displayed(bool, optional): (Default value = False)

    Returns:

    Raises:

    """

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    box_tensor = [torch.stack([torch.tensor(x) for x in boxes]).cuda()]
    
    roialign = RoIAlign(output_shape,spatial_scale=1.0,sampling_ratio=-1)
    image_rois = roialign.forward(image_tensor,box_tensor)

    # None branched syntax
    if displayed:
        [plot.display_3D(x[0],step=1) for x in tensor2numpy(image_rois)]
    return image_rois
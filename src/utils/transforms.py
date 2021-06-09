import torch

class RoiTransform:
    """Apply ROI transform to shange shape of images"""
    
    def __init__(self, output_shape:Tuple[int,int,int]=None, boundary_boxes:Union[List[Tuple[int,int,int,int,int,int]]]=None, batch_size=6,**args):
        """
        Transform boundary boxes to correct format.
        
        Args:
            output_shape (Tuple): Shape that the image input should be transformed to.
            boundary_boxes (List): Availbile boundary boxes. Each target class need at least one boundary box!
            batch_size (int): Input batch size of data
        """
        if not boundary_boxes: raise ValueError("bounding_boxes list can not be empty!")
        
        self.batch_size = batch_size
        self.roi = RoIAlign(output_shape,spatial_scale=1.0,sampling_ratio=-1)
        self.num_bbox = len(boundary_boxes)

        if isinstance(boundary_boxes, list):
            self.boundary_boxes = convert_boxes_to_roi_format([torch.stack([torch.Tensor(x) for x in boundary_boxes])])
        elif isinstance(boundary_boxes, dict):
            self.boundary_boxes = {key:convert_boxes_to_roi_format([torch.stack([torch.Tensor(x) for x in value])]) for key,value in boundary_boxes.items()}
        else:
            raise ValueError("boundary_boxes needs to be of type list or dict")
            
    def __call__(self, x:torch.Tensor, y):
        """
        Expect to take an y of integer type and if boundary_boxes are a dict then the key should be a numeric value.
        
        Args:
            x (Tensor): Input value. Expect shape (B,C,D,H,W)
            y (Tensor): Target value
        """

        if isinstance(self.boundary_boxes, list):
            image_rois = self.roi.forward(x,torch.cat(x.shape[0]*[self.boundary_boxes.to(x.device)]))#.detach()
            y = self.num_bbox*y
        elif isinstance(self.boundary_boxes, dict):
            image_rois = self.roi.forward(x,torch.cat([self.boundary_boxes[one_target].to(x.device) for one_target in tensor2numpy(y)]))#.detach() #x.shape[0]*[self.boundary_boxes[y]

            y = torch.from_numpy(np.concatenate([len(self.boundary_boxes[one_target])*[one_target] for one_target in tensor2numpy(y)])).to(x.device)

        else:
            raise ValueError("boundary_boxes needs to be of type list or dict")

        return image_rois, y
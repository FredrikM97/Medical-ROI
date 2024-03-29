import numpy as np
import pytorch_lightning as pl
import torch
import torchcam

from src.cam import CAM
from src.files.preprocess import tensor2numpy


class CAMCallback(pl.callbacks.Callback):
    """ """
    #https://stackoverflow.com/questions/62494963/how-to-do-class-activation-mapping-in-pytorch-vgg16-model
    
    #def on_fit_start(self, trainer, pl_module) -> None:
        
    #    self.cam_model = CAM(self.cam_type, trainer.model)
        
    def __init__(self, cam_type=torchcam.cams.GradCAMpp):
        """

        Parameters
        ----------
        cam_type :
            (Default value = torchcam.cams.GradCAMpp)

        Returns
        -------

        
        """
        self.cam_type = cam_type
        print("What is the cam type",self.cam_type)
        
    def on_epoch_end(self, trainer, pl_module):
        """

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        #trainer.model.eval()
        self.cam_model = CAM(self.cam_type, trainer.model)
        
        #print(trainer.model.requires_grad)
        image_sample,label = self.get_one_sample(pl_module)
        
        print(image_sample.shape)
        class_scores, class_idx = self.cam_model.evaluate(image_sample)
        
        print("Some other stuff",class_scores)
        fig = self.cam_model.plot(class_scores, class_idx, image_sample, class_label=label)
        
        trainer.logger.experiment.add_figure(f"{self.cam_model.cam.__name__}/{prefix}", fig,trainer.current_epoch)
    
        
    def get_one_sample(self, pl_module):
        """

        Args:
          pl_module: 

        Returns:

        Raises:

        """
        for i, sample in enumerate(pl_module.val_dataloader()):   # Stepping through dataloader might mess up validation elsewhere ?
            # Dont call cuda directly (this is not optimized :( ))
            x,y = sample
            return tensor2numpy(x[0][0]),tensor2numpy(y[0])
            #return sample[0][0, np.newaxis, :, :, :, :].cuda()
import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import skimage.transform

class CAMCallback(pl.callbacks.Callback):
    #https://stackoverflow.com/questions/62494963/how-to-do-class-activation-mapping-in-pytorch-vgg16-model
    display_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224,224)),
    ])
    preprocess = torchvision.transforms.Compose([
       torchvision.transforms.Resize((224,224)),
       torchvision.transforms.ToTensor()
    ])

    def __init__(self, model):
        self.model = model
        #self.get_all_layers()
    
    def on_epoch_end(self, trainer, pl_module):
        trainer.model.eval()
        #self.get_all_layers(self.model)
        self.showCAM(pl_module)
    
    def showCAM(self, pl_module):
        prediction_var = self.get_one_sample(pl_module)
        pred_probabilities, activated_features = self.get_layer_probability(prediction_var, 'conv_layer2')
        weight_softmax = self.get_layer_weights('fc2')
        class_idx = torch.topk(pred_probabilities,1)[1].int()
        
        overlay = self.getCAM(activated_features.features, weight_softmax, class_idx)
        
        grid_img = torchvision.utils.make_grid(prediction_var[0].cpu().detach().permute(1, 0, 2, 3))
        grid_img = grid_img.permute(1, 2, 0).numpy().transpose(1,2,0)

        grid_cad = torchvision.utils.make_grid(overlay[0].cpu().detach(), nrow=5)
        #rid_cad = grid_cad.permute(1, 2, 0).cpu().numpy()
        grid_cad = grid_cad.numpy().transpose(1,2,0)
     
        #plt.imshow(grid_img)
        print("What shape=?",overlay[0].shape, grid_cad.shape)
        #plt.imshow(skimage.transform.resize(grid_cad, prediction_var[0].shape[1:]) , alpha=0.5, cmap='jet')
        plt.imshow(grid_cad, alpha=0.5, cmap='jet')
        #plt.imshow(skimage.transform.resize(overlay[0], img.size), alpha=0.5, cmap='jet');
        plt.show()
        
    def getCAM(self,feature_conv, weight_fc, class_idx):
        _, nc, d, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, d*h*w)))
        #cam = cam.reshape(d, 1, h, w)
        cam = cam.reshape(d, h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = torch.from_numpy(cam)
        return [cam_img]
    
    def get_one_sample(self, pl_module):
        for i, sample in enumerate(pl_module.val_dataloader()):   # Stepping through dataloader might mess up validation elsewhere ?
            # Dont call cuda directly (this is not optimized :( ))
            return sample[0][0, np.newaxis, :, :, :, :].cuda()

            
    def get_all_layers(self, layer):
        for name, layer in layer._modules.items():
            if isinstance(layer, torch.nn.Sequential):
                self.get_all_layers(layer)
            else:
                print(name)
                #layer.register_forward_hook(hook_fn)
    
    def get_layer_probability(self, prediction_var, layer:str):
        final_layer = self.model._modules.get(layer)
        activated_features = SaveFeaturesHook(final_layer)
        
        pred_probabilities  = self.model(prediction_var).data.squeeze()
        activated_features.remove()
        
        return pred_probabilities, activated_features
        
    def get_layer_weights(self, dense_layer):
        weight_softmax_params = list(self.model._modules.get(dense_layer).parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].detach().cpu().data.numpy())
        return weight_softmax

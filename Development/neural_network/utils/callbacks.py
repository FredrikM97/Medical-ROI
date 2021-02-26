from utils.hooks import ActivationMapHook, SaveFeaturesHook
from utils import meanMetric, ROC
import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import skimage.transform

class ActivationMap(pl.callbacks.Callback):
    
    def __init__(self, model):
        self.hooks = []
        for name, module in model.named_modules():
            if type(module) == torch.nn.modules.conv.Conv3d:
                self.hooks.append(ActivationMapHook(module, name))
    
    def on_epoch_end(self, trainer, pl_module):
        set_to_train = False
        if trainer.model.training:
            set_to_train = True
        

        trainer.model.eval()

        for hook in self.hooks:
            hook.register()
        
 
        
        for i, sample in enumerate(pl_module.val_dataloader()):   # Stepping through dataloader might mess up validation elsewhere ?
            # Dont call cuda directly (this is not optimized :( ))
            trainer.model(sample[0][0, np.newaxis, :, :, :, :].cuda())
            break
 
              
        for hook in self.hooks:
            
            hard_filter_limit = 100
            filters_to_use = min(hook.features.shape[1], hard_filter_limit)
            
            width = 10
            height = int(np.ceil(filters_to_use / width))
            size = 1
            slice_index = int(hook.features.shape[4] / 2)
            min_value = 0
            max_value = 1
            #min_value = hook.features[0].min()
            #max_value = hook.features[0].max()
            
            fig, axs = plt.subplots(height, width, figsize = (width * size, height * size))
            plt.subplots_adjust(wspace = 0, hspace = 0)
            fig.suptitle(f"{hook.name} (Showing {filters_to_use}/{hook.features.shape[1]} filters) (slice [:, :{slice_index}]) size [{hook.features.shape[2]}, {hook.features.shape[3]}])")
            
            for i in range(width * height):
                ax = axs[int(i / width), i % width]
                if i < filters_to_use:
                    colorplot = ax.imshow(hook.features[0][i][:, :, slice_index], vmin = min_value, vmax = max_value,alpha=0.5, cmap='jet')#cmap = 'viridis')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.set_visible(False)
                    
            fig.subplots_adjust(right = 0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(colorplot, cax = cbar_ax)

            trainer.logger.experiment.add_figure("featuremap",fig,trainer.current_epoch) 
            #fig.show()
            
            # https://www.tensorflow.org/tensorboard/image_summaries
            # Save to buffer, write to tb
            
        for hook in self.hooks:
            hook.unregister()
            
        if set_to_train:
            trainer.model.train()
        
            
class MetricCallback(pl.callbacks.Callback):
    
    # TODO: Add so model store predicted and target. Then we can reuse that instead of external stuff..
    def __init__(self, num_classes=3):
        self.num_classes = 3
        self.train_loss = meanMetric(compute_on_step=False).cuda()
        
        self.val_cm = pl.metrics.ConfusionMatrix(num_classes, compute_on_step=False).cuda()
        self.val_accuracy = pl.metrics.Accuracy(compute_on_step=False).cuda()
        self.val_loss = meanMetric(compute_on_step=False).cuda()
        self.val_roc = pl.metrics.ROC(num_classes=num_classes, compute_on_step=False).cuda()
        
    def on_train_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):  
        self.train_loss(outputs[0][0]['minimize'])# This is loss?
    
    def on_validation_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pred = outputs['predicted/val']
        targ = outputs['target/val']
        loss = outputs['loss/val']
        prob = outputs['probability/val']
        self.val_cm(pred, targ)
        self.val_accuracy(pred, targ)
        self.val_loss(loss)
        self.val_roc(prob, targ)
        #print("asdads", outputs['probability/val'], outputs['target/val'])
        
    def on_epoch_end(self, trainer, pl_module):
        train_loss = self.train_loss.compute()
        self.add_scalar(trainer, train_loss, metric_prefix='loss',prefix='train')
 
        self.custom_histogram_adder(trainer)
        val_cm = self.val_cm.compute()
        val_acc = self.val_accuracy.compute()
        val_loss = self.val_loss.compute()
        val_roc = self.val_roc.compute()
        
        self.cm_plot(trainer, val_cm, prefix='val')
        ROC(trainer, val_roc, prefix='val')
        self.add_scalar(trainer, val_acc, metric_prefix='accuracy',prefix='val')
        self.add_scalar(trainer, val_loss, metric_prefix='loss',prefix='val')
        
        # Do this last since it seem to be very buggy but useful. Note: we add hparam so it does not try to overwrite existing data
        trainer.logger.log_hyperparams(
            dict(pl_module.hparams), {
            f'hparam/loss/train':train_loss,
            f'hparam/loss/val':val_loss,
            f'hparam/accuracy/val':val_acc,
        })
  
    def cm_plot(self, trainer, cm, prefix=''):
     
        fig=plt.figure();
        ax = sns.heatmap(cm.cpu().numpy(), annot=True, annot_kws={"size": 12})
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        trainer.logger.experiment.add_figure(f"confmat/{prefix}", fig,trainer.current_epoch)
    
    def add_scalar(self, trainer, metric, metric_prefix=None,prefix=''):
        assert metric_prefix
        trainer.logger.experiment.add_scalar(f"{metric_prefix}/{prefix}",
                                            metric,
                                            trainer.current_epoch)
        
    def custom_histogram_adder(self, trainer):
        # iterating through all parameters
        for name,params in trainer.model.named_parameters():
            trainer.logger.experiment.add_histogram(name,params,trainer.current_epoch)
            
class CAM(pl.callbacks.Callback):
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

from collections import OrderedDict
from models.base_model import BaseModel
from optimizers.adam_optimizer import Adam
import random
import torch
from sklearn.metrics import accuracy_score
import sys
from models.architectures import UNet3D


class Segmentation3DModel(BaseModel):
    def __init__(self, configuration):
        """Initialize the model.
        """
        super().__init__(configuration)

        #self.loss_names = ['segmentation']
        self.network_names = ['unet']

        self.netunet = UNet3D(1, 3)
        self.netunet = self.netunet.to(self.device)
        
        if self.is_train:  # only defined during training time
            self.criterion_loss = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.netunet.parameters(), lr=configuration['lr'])
            self.optimizers = [self.optimizer]

        # storing predictions and labels for validation
        self.val_predictions = []
        self.val_labels = []
        self.val_images = []
        
        self.compile_metrics(self.metrics)

    def forward(self):
        """Run forward pass.
        """
        self.output = self.netunet(self.input)


    def backward(self):
        """Calculate losses; called in every training iteration.
        """

        self.loss = self.criterion_loss(self.output, self.label)
        self.compiled_loss.append(self.loss.item())
        
    def optimize_parameters(self):
        """Calculate gradients and update network weights.
        """
        self.loss.backward() # calculate gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()


    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        #self.val_images.append(self.input)
        #self.val_predictions.append(self.output)
        #self.val_labels.append(self.label)
        
        self.update_metrics(self.val_metrics, self.output, self.input)

    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['accuracy'] = val_accuracy

        visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []
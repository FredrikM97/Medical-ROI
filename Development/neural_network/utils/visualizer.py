import numpy as np
import sys
from subprocess import Popen, PIPE
import utils
#import visdom


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    
def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break
        pass
class Visualizer():
    """This class includes several functions that can display images and print logging information.
    """

    def __init__(self, configuration):
        """Initialize the Visualizer class.
        Input params:
            configuration -- stores all the configurations
        """
        self.configuration = configuration  # cache the option
        self.display_id = 0
        self.name = configuration['name']

        self.ncols = 0
    def reset(self):
        """Reset the visualization.
        """
        pass


    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at the default port.
        """
        pass


    def plot_current_losses(self, epoch, counter_ratio, losses):
        """Display the current losses on visdom display: dictionary of error labels and values.
        Input params:
            epoch: Current epoch.
            counter_ratio: Progress (percentage) in the current epoch, between 0 to 1.
            losses: Training losses stored in the format of (name, float) pairs.
        """
        pass


    def plot_current_validation_metrics(self, epoch, metrics):
        """Display the current validation metrics on visdom display: dictionary of error labels and values.
        Input params:
            epoch: Current epoch.
            losses: Validation metrics stored in the format of (name, float) pairs.
        """
        pass


    def plot_roc_curve(self, fpr, tpr, thresholds):
        """Display the ROC curve.
        Input params:
            fpr: False positive rate (1 - specificity).
            tpr: True positive rate (sensitivity).
            thresholds: Thresholds for the curve.
        """
        pass


    def show_validation_images(self, images):
        """Display validation images. The images have to be in the form of a tensor with
        [(image, label, prediction), (image, label, prediction), ...] in the 0-th dimension.
        """
        pass


    def print_current_losses(self, epoch, max_epochs, iter, max_iters, losses):
        """Print current losses on console.
        Input params:
            epoch: Current epoch.
            max_epochs: Maximum number of epochs.
            iter: Iteration in epoch.
            max_iters: Number of iterations in epoch.
            losses: Training losses stored in the format of (name, float) pairs
        """
        pass
    
    
    def print_model(self):
        pass
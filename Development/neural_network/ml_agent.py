import argparse
import math
import time
import multiprocessing

from datasets import create_dataset
from configs import load_config
from models import create_model
from utils.visualizer import Visualizer

"""Performs training of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
    export: Whether to export the final model (default=True).
"""
class Agent:
    def __init__(self, config_name:str, export:bool=True):
        #multiprocessing.set_start_method('spawn', True)
        # Setup config
        print('Setup configurations...')
        self.config = load_config(config_name)
        
        # Setup model
        self.model = create_model(self.config['model_params'])
        self.model.setup()
        
        # Setup dataloader
        self.train_dataset = create_dataset(self.config['train_dataset_params'])
        self.val_dataset = create_dataset(self.config['val_dataset_params'])
        
        # Setup visualisation
        self.visualizer = Visualizer(self.config['visualization_params'])  
        
        
        # Other meta
        
    def fit(self):
        # Setup epochs 
        starting_epoch = self.config['model_params']['load_checkpoint'] + 1
        num_epochs = self.config['model_params']['max_epochs']

        # Setup meta
        print_freq = self.config['printout_freq']
        model_update_freq = self.config['model_update_freq']


        print("Start training...")
        # Run through dataset num_epochs number of times.
        for epoch in range(starting_epoch, num_epochs):
            epoch_start_time = time.time()  # timer for entire epoch
            self.pre_epoch_callback(epoch)

    
            # Train model
            self.model.train()
            for step, data in enumerate(self.train_dataset,0):  # inner loop within one epoch
                self.train_step(data)
                self.post_step_callback(epoch)
            
            # Evaluate model
            self.evaluate()

            print('Saving model at the end of epoch {0}'.format(epoch))
            self.model.save_networks(epoch)
            self.model.save_optimizers(epoch)
            self.model.update_learning_rate() # update learning rates every epoch
            
            self.post_epoch_callback(epoch)
            
            print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch, num_epochs, time.time() - epoch_start_time), sep="\r")
            
            
        if export: self.export()
            

        #return self.model.get_hyperparam_result()

    def train_step(self, data):
        "Training model by enumerate one item of mini batch"
        self.model.set_input(data)         # unpack data from dataset and apply preprocessing
        self.model.forward()
        self.model.backward()
        #if i % model_update_freq == 0:
        self.model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        
        return model.get_current_losses()

    def test_step(self, data):
        "Testing model by enumerate one item of mini batch"
        # Validate model
        self.model.set_input(data)
        self.model.test()
    
    def post_step_callback(self, epoch):
        "Do something when step is finished"
        self.update_metrics(self.metrics, self.output, self.label)
        print('[%d/%d][%d/%d]\tLoss%s\tMetrics%s: ' % (epoch, num_epochs, step, len(self.train_dataset),  self.model.loss.item(), f'{k}:{v}'.join(self.model.compiled_metrics.items())), sep="\r")
        
    def pre_epoch_callback(self, epoch):
        "Do something before epoch starts"
        self.train_dataset.dataset.pre_epoch_callback(epoch)
        self.model.pre_epoch_callback(epoch)
    
    def post_epoch_callback(self, epoch):
        "Do something after epoch"
        self.model.post_epoch_callback(epoch, self.visualizer)
        self.train_dataset.dataset.post_epoch_callback(epoch)
    
    def export_model(self):
        print('Exporting model')
        self.model.eval()
        custom_config = self.config['train_dataset_params']
        custom_config['loader_params']['batch_size'] = 1 # set batch size to 1 for tracing
        dl = self.train_dataset.get_custom_dataloader(custom_config)
        sample_input = next(iter(dl)) # sample input from the training dataset
        self.model.set_input(sample_input)
        self.model.export()
    
    def evaluate(self):
        self.pre_eval_callback()

        self.model.eval()
        for step, data in enumerate(self.val_dataset,0):
            self.test_step(data)
            self.post_step_callback(epoch)

        self.post_eval_callback()
        
    """
    def some_odd_visualizer_function(self):
        print('The number of samples:\
              \n\tTraining: {0}\
              \n\tValidation:{1}\
              '.format(len(self.train_dataset),len(self.val_dataset)))
              
        #train_iterations = len(train_dataset)
        #train_batch_size = config['train_dataset_params']['loader_params']['batch_size']
        #visualizer.reset()
        if i % print_freq == 0:
            losses = model.get_current_losses()
            #visualizer.print_current_losses(epoch, num_epochs, i, math.floor(train_iterations / train_batch_size), losses)
            #visualizer.plot_current_losses(epoch, float(i) / math.floor(train_iterations / train_batch_size), losses)
    """
    
"""
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
"""
import argparse
from datasets import create_dataset
from utils import parse_configuration
import math
from models import create_model
import time
from utils.visualizer import Visualizer
import multiprocessing

"""Performs training of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
    export: Whether to export the final model (default=True).
"""
def train(config_file, export=True):
    multiprocessing.set_start_method('spawn', True)

    print('Setup configurations...')
    config = parse_configuration(config_file)
    
    # Setup model
    model = create_model(config['model_params'])
    model.setup()
    
    # Setup dataloader
    train_dataset = create_dataset(config['train_dataset_params'])
    val_dataset = create_dataset(config['val_dataset_params'])
    
    print('The number of samples:\
          \n\tTraining: {0}\
          \n\tValidation:{1}\
          '.format(len(train_dataset),len(val_dataset)))
    
    # Setup visualisation
    visualizer = Visualizer(config['visualization_params'])   # create a visualizer that displays images and plots
    
    # Setup epochs 
    starting_epoch = config['model_params']['load_checkpoint'] + 1
    num_epochs = config['model_params']['max_epochs']
    
    # Setup meta
    print_freq = config['printout_freq']
    model_update_freq = config['model_update_freq']
    # Init meta
    steps = 0
    running_loss = 0
    

    print("Start training...")
    # Run through dataset num_epochs number of times.
    for epoch in range(starting_epoch, num_epochs):
        epoch_start_time = time.time()  # timer for entire epoch
        train_dataset.dataset.pre_epoch_callback(epoch)
        model.pre_epoch_callback(epoch)

        #train_iterations = len(train_dataset)
        #train_batch_size = config['train_dataset_params']['loader_params']['batch_size']
        
        # Train model
        model.train()
        for i, data in enumerate(train_dataset,0):  # inner loop within one epoch
            steps += 1
            inputs, labels = data
            print("type of ", type(inputs), type(labels))
            #visualizer.reset()
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.forward()
            model.backward()

            if i % model_update_freq == 0:
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if i % print_freq == 0:
                losses = model.get_current_losses()
                #visualizer.print_current_losses(epoch, num_epochs, i, math.floor(train_iterations / train_batch_size), losses)
                #visualizer.plot_current_losses(epoch, float(i) / math.floor(train_iterations / train_batch_size), losses)
        
        # Validate model
        model.eval()
        for i, data in enumerate(val_dataset):
            model.set_input(data)
            model.test()
        
        
        
        model.post_epoch_callback(epoch, visualizer)
        train_dataset.dataset.post_epoch_callback(epoch)

        print('Saving model at the end of epoch {0}'.format(epoch))
        model.save_networks(epoch)
        model.save_optimizers(epoch)

        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch, num_epochs, time.time() - epoch_start_time), sep="\r")

        model.update_learning_rate() # update learning rates every epoch

    if export:
        print('Exporting model')
        model.eval()
        custom_config = config['train_dataset_params']
        custom_config['loader_params']['batch_size'] = 1 # set batch size to 1 for tracing
        dl = train_dataset.get_custom_dataloader(custom_config)
        sample_input = next(iter(dl)) # sample input from the training dataset
        model.set_input(sample_input)
        model.export()

    return model.get_hyperparam_result()

    
"""
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
"""
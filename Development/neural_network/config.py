from dataclasses import dataclass

@dataclass
class LoggerConfig('NNConfig'):
    # All configs related to the logger (paths etc..)
    
    DIR_TEXT:str=base_dir+'/text/' + RUN_TIME
    DIR_IMAGE:str=base_dir+'/image/' + RUN_TIME
    DIR_HPARAM:str=base_dir+'/hparam/' + RUN_TIME
    DIR_TRAIN:str= base_dir+'/train/' + RUN_TIME
    DIR_VALIDATION:str=base_dir+'/validation/' + RUN_TIME
    

@dataclass
class NNConfig:
    # All configs related to the neural network (input, sizes, loss etc.).
    RUN_TIME:str # Define which run it is 
    DIR_BASE:str = os.getcwd()
    DIR_LOGS:str = DIR_BASE +'/logs/nn'
    DIR_DATA:str = DIR_BASE + '/data'
        
    DIR_DATA = "../data/processed"

    DIR_CHECKPOINT:str=DIR_BASE + "/checkpoint/" + RUN_TIME
        
    GPU:int = 0                 # GPU ID
    DROPOUT_PROB:int = 0.5      # Probability to keep a node in dropout
    IMAGE_WIDTH:int = 32       # image width
    IMAGE_HEIGHT:int = 32      # image height
    IMAGE_CHANNEL:int = 3       # image channel
    NUM_EPOCHS:int = 5        # epoch number
    BATCH_SIZE:int = 30         # batch size
    SEQUENCE_LENGTH:int = 10         # length of each sequence
    LEARNING_RATE:int = 0.01  # learning rate
    LR_DECAY_FACTOR:int = 0.1   # multiply the learning rate by this factor
    PRINT_EVERY:int = 20        # print in every 50 epochs
    SAVE_EVERY:int = 1         # save after each epoch
    DEBUG_MODE:int = True      # print log to console in debug mode
    DATA_AUGMENTATION:int = True   # Whether to do data 
    LOSS_FN:int
    OPTIMIZER:int
        


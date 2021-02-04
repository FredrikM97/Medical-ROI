from configs import load_config
from models import create_model
from datasets import create_dataset
from pytorch_lightning.callbacks import ModelCheckpoint,ProgressBar
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

class Agent:
    def __init__(self, config_name:str, export:bool=True):

        print('Setup configurations...')
        # ****** Setup configurations ******
        self.config = load_config(config_name)
        
        
        # ****** Setup loggers ******
       
        checkpoint_callback = ModelCheckpoint(
            #filename=self.config['model_params']['checkpoint_path']+'mnist_{epoch}-{val_loss:.2f}',
            #monitor='val_loss', 
            #mode='min', save_top_k=3
        )
        progressbar_callback = ProgressBar()
      
        #lr_logger = LearningRateLogger()
        tb_logger = pl_loggers.TensorBoardLogger("logs/tb_logs", name=self.config['model_params']['model_name'])
        
        # ****** Setup model ******
        self.model = create_model(self.config['model_params'])
        
        # ****** Setup dataloader ******
        dataset = create_dataset(self.config['dataset_params'])
        
        self.train_loader = dataset.train_dataloader()
        self.val_loader = dataset.val_dataloader()
        
        print(f"Size of:\n\tTrainset: {len(self.train_loader)}\n\tValset: {len(self.val_loader)}")
        
        # ****** Setup trainer ******
        
        self.trainer = pl.Trainer(
            max_epochs=self.config['model_params']['max_epochs'], 
            profiler="simple", 
            #callbacks=[lr_logger], 
            #gpus=None, 
            logger=tb_logger,
            callbacks=[checkpoint_callback,progressbar_callback],
            #default_root_dir=self.config['model_params']['checkpoint_path']
        ) #gpus=1
        
        
    
        
        
    def fit(self):
        self.trainer.fit(
            self.model, 
            train_dataloader=self.train_loader, 
            val_dataloaders=self.val_loader
        )
    
    def test(self):
        trainer.test(
            test_dataloaders=self.test_dataloader
        )
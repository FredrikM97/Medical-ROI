class Settings_tensorboard:
    pass

class settings_logger:
    pass

class logger:
    _logdir=None
    _enabled=True
    _tensorboard=None
    
    def __init__(self, path):
        self._logdir = path
    
    def enable(self):
        "Enable logger"
        self._enabled = True
    
    def disable(self):
        "Disable logger"
        self._enabled = False
    
    def hash_filename(self):
        "Create unique hash value for logger."
        pass
    
    def to_tensorboard(self):
        "Send data to tensorboard"
        pass
    
    def tensorboard(self):
        "Create tensorboard"
        pass
    
    def save_image(self, img):
        "Save image to log dir"
        pass
    
    def save_text(self,txt):
        "Save text to log dir"
        pass
    
    
        
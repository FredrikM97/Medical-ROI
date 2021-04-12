import logging
import logging.config
from pathlib import Path
from src.utils.load import load_json
import sys
import logging

def getLogger(name, fmt="\x1b[80D\x1b[1A\x1b[K%(message)s",terminator='\n'):#"[%(asctime)s]%(name)s<%(levelname)s>%(message)s"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    #logger.flush = sys.stdout.flush
    
    cHandle = logging.StreamHandler()
    cHandle.terminator = terminator
    
    cHandle.setFormatter(logging.Formatter(fmt=fmt, datefmt="%H:%M:%S"))
    logger.addHandler(cHandle)
    return logger


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = load_json(log_config)
        
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
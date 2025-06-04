
from pathlib import Path, PurePath
import logging, os

def setup_logging(name:str):
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO)
    return logging.getLogger(name)

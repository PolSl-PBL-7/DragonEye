import logging
import os
from pathlib import Path

def initialize_logger(output_dir, args):
    logs_dir = Path(output_dir) / 'logs' 
    os.makedirs(logs_dir, exist_ok = True)
    logging.basicConfig(filename=logs_dir / 'logs.log',
                        level = logging.INFO,
                        filemode="w",
                        format = "%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                        )
    for arg, value in vars(args).items():
        logging.info(f"Argument {arg}: {value}")


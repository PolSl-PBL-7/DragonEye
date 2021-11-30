import logging
import os
from pathlib import Path


def initialize_logger(output_dir, args_dict):
    """
    Args:

    args_dict (dict of dicts): dict in form
    {
        "config name" : dict_with_config,
        "other config" : dict with other config
    }

    """
    logs_dir = Path(output_dir) / 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(filename=logs_dir / 'logs.log',
                        level=logging.INFO,
                        filemode="w",
                        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                        )
    for arg_dict_name, arg_dict in args_dict.items():
        for arg, value in arg_dict.items():
            logging.info(f"{arg_dict_name} - Argument {arg}: {value}")

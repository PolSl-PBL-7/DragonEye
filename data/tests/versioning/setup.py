from os import path

from data.versioning.configs import wandb_config
from data.versioning.handler import WandbDatasetHandler


class TestSetup:
    def __init__(self):
        pass

    def setup_dataset(self, dataset_path='./data'):
        """
        Initial setup of the datasets required for testing purposes.
        Parameters
        ----------
        dataset_path :
        Path of the destination directory of the dataset
        Returns
        -------

        """

        if path.exists(dataset_path): pass
        else:
            handler = WandbDatasetHandler()
            config = wandb_config
            config = config._replace(path=dataset_path)
            handler.load_dataset(dataset_config=config)

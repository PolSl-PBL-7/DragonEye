from abc import ABC, abstractmethod

import wandb

from data.versioning.configs import DatasetConfig


class DatasetHandler(ABC):
    """
    Abstract class for loading and saving dataset
    """
    @abstractmethod
    def save_dataset(self, dataset_config):
        pass

    @abstractmethod
    def load_dataset(self, dataset_config):
        pass


class WandbDatasetHandler(DatasetHandler):

    """
    Class saving and loading datasets from Weights and biases
    """

    def save_dataset(self, dataset_config):
        pass

    def download_dataset(self, dataset_config):
        """
        Method responsible for retrieving given dataset from Weights and Biases' storage

        Parameters
        ----------
        dataset_config (namedtuple) : Names tuple storing configuration needed to load the dataset from Weights and Biases.

        """

        wandb.init(project=dataset_config.project_name, entity=dataset_config.entity)
        run = wandb.init(job_type=dataset_config.job_type)
        artifact = run.use_artifact(f'{dataset_config.dataset_name}:latest')
        artifact.download(dataset_config.path)
        return 0


if __name__ == "__main__":
    db = WandbDatasetHandler()
    config = DatasetConfig("avenue-experiments", "polsl-pbl-7", "test", "avenue-dataset", "./dataset")
    print(config)
    db.load_dataset(config)

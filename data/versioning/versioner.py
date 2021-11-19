from abc import ABC, abstractmethod
from typing import Tuple, NamedTuple, Optional, Union
from pathlib import  Path

import wandb


class VersioningConfig(NamedTuple):
    project_name: str = 'avenue-experiments'
    entity: str = 'polsl-pbl-7'
    job_type: str = 'test'
    dataset_name: str = 'avenue-dataset'
    dataset_path: Union[str, Path] = './dataset'
    type: str = 'folder'
    tag: str = 'latest'
    artifact_type: str = 'dataset'
    experiment_name: str = 'test'


class DatasetVersioner(ABC):
    """
    Abstract class for loading and saving dataset
    """

    @abstractmethod
    def save_dataset(self, dataset_config):
        pass

    @abstractmethod
    def load_dataset(self, dataset_config):
        pass


class WandbDatasetVersioner(DatasetVersioner):
    """
    Class saving and loading datasets from Weights and biases
    """

    def save_dataset(self, dataset_config: VersioningConfig):
        """
        Method responsible for uploading given dataset to Weights and Biases' storage

        Parameters
        ----------
        dataset_config (namedtuple) : Names tuple storing configuration needed to load the dataset from Weights and Biases.

        """
        run = wandb.init(project=dataset_config.project_name, entity=dataset_config.entity,
                         name=dataset_config.experiment_name, job_type=dataset_config.job_type)
        artifact = wandb.Artifact(dataset_config.dataset_name, type=dataset_config.artifact_type)

        if dataset_config.type == 'folder':
            artifact.add_dir(str(dataset_config.dataset_path))

        elif dataset_config.type == 'file':
            artifact.add_file(str(dataset_config.dataset_path))

        run.log_artifact(artifact)
        run.finish()
        wandb.finish()

    def load_dataset(self, dataset_config: VersioningConfig):
        """
        Method responsible for retrieving given dataset from Weights and Biases' storage

        Parameters
        ----------
        dataset_config (namedtuple) : Names tuple storing configuration needed to load the dataset from Weights and Biases.

        """

        run = wandb.init(project=dataset_config.project_name, entity=dataset_config.entity,
                         name=dataset_config.experiment_name, job_type=dataset_config.job_type)
        artifact = run.use_artifact(f'{dataset_config.dataset_name}:{dataset_config.tag}')
        artifact.download(str(dataset_config.dataset_path))
        run.finish()
        wandb.finish()




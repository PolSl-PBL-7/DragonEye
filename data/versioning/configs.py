from collections import namedtuple


DatasetConfig = namedtuple("DatasetConfig", "project_name entity job_type dataset_name path")

wandb_config = DatasetConfig("avenue-experiments", "polsl-pbl-7", "test", "avenue-dataset", "./dataset")

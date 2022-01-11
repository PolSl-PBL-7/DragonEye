import wandb
import os.path as path
import pprint
import sys

from pipelines import pipeline_runner as pr


# Configuration
PROJECT = 'testing-sweep'
NUM_RUNS = 5

sweep_config = {
    'name': 'test-sweep',
    'method': 'grid',
    'parameters': {
        'epochs': {
            'values': [1, 2, 5]
        },
        'batch_size': {
            'values': [2, 4, 8]
        },
        'fps': {
            'values': [3, 5, 10]
        },
        'time_window': {
            'values': [3, 5, 10]
        }
    }
}

def train_sweep(config_defaults = None) -> None:
    pprint.pprint(wandb.init())
    config = wandb.config
    config_dict = pr.build_config_dict(path.join('.', 'pipelines', 'configs'))

    pipeline = pr.get_pipeline_by_type(pr.PipelineType.training_pipeline)

    config_dict['training_params']['source_arams']['fps'] = config['fps']
    config_dict['training_params']['processor_params']['batch_size'] = config['batch_size']
    config_dict['training_params']['processor_params']['time_window'] = config['time_window']
    config_dict['training_params']['training_params']['epochs'] = config["epochs"]

    pipeline(**config_dict)

if __name__ == '__main__':
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    wandb.agent(sweep_id, function=train_sweep, count=NUM_RUNS)

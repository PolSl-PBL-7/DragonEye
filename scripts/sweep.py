import wandb
import os.path as path
import pprint

from pipelines import pipeline_runner as pr
from utils.file_loaders import Json


# Configuration
PROJECT = 'testing-sweep'
NUM_RUNS = 5
CONFIG_TEMPLATE_LOCATION = path.join('.', 'pipelines', 'configs', 'training_params.json')

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


def train_sweep(config_defaults=None) -> None:
    wandb.init()
    config = wandb.config

    if path.isfile(CONFIG_TEMPLATE_LOCATION):
        config_dict = Json.load(CONFIG_TEMPLATE_LOCATION)
    else:
        config_dict = pr.build_config_dict(CONFIG_TEMPLATE_LOCATION)['training_params']

    pipeline = pr.get_pipeline_by_type(pr.PipelineType.training_pipeline)

    print(10 * '=', 'TEMPLATE', 10 * '=')
    pprint.pprint(config_dict)
    print(10 * '=', 'EOF', 10 * '=')

    config_dict['source_params']['fps'] = config['fps']
    config_dict['processor_params']['batch_size'] = config['batch_size']
    config_dict['processor_params']['time_window'] = config['time_window']
    config_dict['training_params']['epochs'] = config["epochs"]

    print(10 * '=', 'RECONFIGURED', 10 * '=')
    pprint.pprint(config_dict)
    print(10 * '=', 'EOF', 10 * '=')

    pipeline(**config_dict)


if __name__ == '__main__':
    if not path.exists(CONFIG_TEMPLATE_LOCATION):
        raise FileNotFoundError('given config file/directory does not exist')

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    wandb.agent(sweep_id, function=train_sweep, count=NUM_RUNS)

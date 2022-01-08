import wandb
import os.path as path

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
        }
    }
}

wandb.login()

def train(epochs) -> None:
    config_dict = pr.build_config_dict(path.join('.', 'pipelines', 'configs'))

    # TODO: modify config

    pipeline = pr.get_pipeline_by_type(pr.PipelineType.training_pipeline)
    
    for e in epochs:
        print(f'TODO: should replace epochs with {e}')
        print(config_dict)
        pipeline(**config_dict)

def sweep_train(config_defaults = None) -> None:
    train([1,2,4])

sweep_id = wandb.sweep(sweep_config, project=PROJECT)

wandb.agent(sweep_id, function=sweep_train, count=NUM_RUNS)

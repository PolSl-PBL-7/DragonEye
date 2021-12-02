from enum import Enum
from wandb.keras import WandbCallback


class CallbackName(Enum):

    wandb_training_loss = 'wandb_training_loss_callback'


def get_callback_by_name(name):
    if name == CallbackName.wandb_training_loss.value:
        return WandbCallback(
            monitor='loss'
        )
    else:
        raise ValueError(f'Callback {name} not found')

from enum import Enum
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping



class CallbackName(Enum):

    wandb_training_loss = 'wandb_training_loss_callback'
    ten_epoch_stop_callback = 'ten_epoch_stop_callback'

def get_callback_by_name(name):
    if name == CallbackName.wandb_training_loss.value:
        return WandbCallback(
            monitor='loss'
        )
    elif name == CallbackName.ten_epoch_stop_callback.value:
        return EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        )
    else:
        raise ValueError(f'Callback {name} not found')

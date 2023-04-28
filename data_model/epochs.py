import config
import segmentation_models_pytorch as smp

from data_model import test_dataloader
from utils import load_model

train_epoch = smp.utils.train.TrainEpoch(
    config.model,
    loss=config.loss,
    metrics=config.metrics,
    optimizer=config.optimizer,
    device=config.DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    config.model,
    loss=config.loss,
    metrics=config.metrics,
    device=config.DEVICE,
    verbose=True,
)


def evaluate_test_data():
    test_epoch = smp.utils.train.ValidEpoch(
        model=load_model(),
        loss=config.loss,
        metrics=config.metrics,
        device=config.DEVICE,
    )
    # evaluate model on test data
    test_logs = test_epoch.run(test_dataloader)
    return test_logs

import config
import segmentation_models_pytorch as smp

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

test_epoch = smp.utils.train.ValidEpoch(
    model=config.get_best_model(),
    loss=config.loss,
    metrics=config.metrics,
    device=config.DEVICE,
)
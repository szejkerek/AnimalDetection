import os

import segmentation_models_pytorch as smp
import torch

from config import CLASSES
from utils import config_line

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d'
lr = 0.0001

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    in_channels=3,
    activation=ACTIVATION,
)

loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    # smp.utils.metrics.Fscore(threshold=0.5),
    # smp.utils.metrics.Accuracy(threshold=0.5),
    # smp.utils.metrics.Recall(threshold=0.5),
    # smp.utils.metrics.Precision(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=lr),
])

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


def save_config(current_path=""):
    f = open(os.path.join(current_path, "config.cfg"), "w")

    f.write(config_line("ENCODER", ENCODER))
    f.write(config_line("ENCODER_WEIGHTS", ENCODER_WEIGHTS))
    f.write(config_line("ACTIVATION", ACTIVATION))
    f.write(config_line("Model", model.__module__))
    f.write(config_line("LossFunction", loss.__module__))
    f.write(config_line("OptimizerFunction", optimizer.__module__))
    f.write(config_line("LearningRate", lr))
    f.write("#Metrics\n")
    for i in range(len(metrics)):
        f.write(config_line("Metric" + str(i) + "_name", metrics[i].__name__))
        f.write(config_line("Metric" + str(i) + "_threshold", metrics[i].threshold))

    f.close()

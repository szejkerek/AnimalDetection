import os

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import DeepLabV3Plus
import torch

from config import COLORS
from utils import config_line

CLASSES = ['animal', 'maskingbackground', 'nonmaskingbackground', 'nonmaskingforegroundattention']
WEIGHTS = torch.tensor([10, 2, 2, 1])

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d'
lr = 0.0001
BATCH_SIZE = 10


model = DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    in_channels=3,
    activation=ACTIVATION,
)

loss = smp.utils.losses.CrossEntropyLoss(weight=WEIGHTS)


metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    #smp.utils.metrics.Fscore(threshold=0.5),
    # smp.utils.metrics.Accuracy(threshold=0.5),
    # smp.utils.metrics.Recall(threshold=0.5),
    # smp.utils.metrics.Precision(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=lr),
])

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


def calculate_score(logs):
    return logs['iou_score']


def calculate_loss(logs):
    return logs['cross_entropy_loss']


def save_config(current_path=""):
    f = open(os.path.join(current_path, "config.cfg"), "w")

    for i in range(len(CLASSES)):
        f.write("#Class_{}\n".format(i))
        f.write(config_line("Name", str(CLASSES[i])))
        f.write(config_line("Color", str(COLORS[i])))
        f.write(config_line("LossWeight", str(WEIGHTS[i])))
        f.write("\n")

    for i in range(len(metrics)):
        f.write("#Metric_{}\n".format(str(i)))
        f.write(config_line("Name", metrics[i].__name__))
        f.write(config_line("Threshold", metrics[i].threshold))
        f.write("\n")

    f.write(config_line("Encoder", ENCODER))
    f.write(config_line("EncoderWeights", ENCODER_WEIGHTS))
    f.write(config_line("Activation", ACTIVATION))
    f.write(config_line("Model", model.__module__))
    f.write(config_line("LossFunction", loss.__class__))
    f.write(config_line("OptimizerFunction", optimizer.__module__))
    f.write(config_line("LearningRate", lr))
    f.write(config_line("BatchSize", BATCH_SIZE))

    f.close()

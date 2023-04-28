import segmentation_models_pytorch as smp
import torch

from config import CLASSES

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d'

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
    dict(params=model.parameters(), lr=0.0001),
])

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


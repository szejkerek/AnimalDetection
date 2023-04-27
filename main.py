from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
import ssl
import segmentation_models_pytorch.utils.metrics as smpu
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context
DATA_DIR = './data'

# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('Loading data...')
    os.system('git clone https://github.com/szejkerek/AnimalClassificationNeuralNetwork ./data')
    print('Done!')

# Paths to datasets
x_train_dir = os.path.join(DATA_DIR, 'GDTrain/Images')
y_train_dir = os.path.join(DATA_DIR, 'GDTrain/Masks')

x_valid_dir = os.path.join(DATA_DIR, 'GDValidation/Images')
y_valid_dir = os.path.join(DATA_DIR, 'GDValidation/Masks')

x_test_dir = os.path.join(DATA_DIR, 'GDTest/Images')
y_test_dir = os.path.join(DATA_DIR, 'GDTest/Masks')


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def crop_to_divisible_by_32(image):
    h, w = image.shape[:2]
    h_new = h - h % 32
    w_new = w - w % 32
    cropped_image = image[:h_new, :w_new]
    return cropped_image


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['animal', 'maskingbackground', 'nonmaskingbackground', 'nonmaskingforegroundattention', 'unlabelled']
    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255), (0, 0, 0)]

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ImagesIds = sorted(os.listdir(images_dir))
        self.MasksIds = sorted(os.listdir(masks_dir))
        self.ids = self.MasksIds
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ImagesIds]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.MasksIds]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = crop_to_divisible_by_32(image)
        mask = crop_to_divisible_by_32(mask)

        # Get the width and height of the image
        height, width, _ = image.shape

        # Calculate the new width and height that are divisible by 32
        new_width = (width // 32) * 32
        new_height = (height // 32) * 32

        # Crop the image to the new width and height
        cropped_image = image[0:new_height, 0:new_width]

        # extract certain classes from mask (e.g. cars)
        masks = []
        for cls_value in self.class_values:
            color = self.COLORS[cls_value]
            newMask = cv2.inRange(mask, color, color)
            masks.append(newMask)

        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


ENCODER = 'vgg16'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['animal', 'maskingbackground', 'nonmaskingbackground', 'nonmaskingforegroundattention', 'unlabelled']
ACTIVATION = 'softmax'
DEVICE = 'cuda'

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    in_channels=3,
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

lossWeights = torch.tensor([0.6, 0.1, 0.1, 0.1, 0.1])
loss = smp.utils.losses.CrossEntropyLoss()
metrics = [
    smpu.IoU(threshold=0.6),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# Lets look at data we have
dataset = Dataset(x_train_dir, y_train_dir,
                  classes=['animal', 'maskingbackground', 'nonmaskingbackground', 'nonmaskingforegroundattention',
                           'unlabelled'])

for i in range(0):
    image, mask = dataset[i]  # get some sample
    visualize(
        image=image,
        animals=mask[..., 0].squeeze(),
        maskingbackground=mask[..., 1].squeeze(),
        nonmaskingbackground=mask[..., 2].squeeze(),
        nonmaskingforegroundattention=mask[..., 3].squeeze(),
        unlabelled=mask[..., 4].squeeze(),
    )

# train model for 40 epochs

max_score = 0

for i in range(0, 30):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# # load best saved checkpoint
# best_model = torch.load('./best_model.pth')
#
# # create test dataset
# test_dataset = Dataset(
#     x_test_dir,
#     y_test_dir,
#     augmentation=get_validation_augmentation(),
#     preprocessing=get_preprocessing(preprocessing_fn),
#     classes=CLASSES,
# )
#
# test_dataloader = DataLoader(test_dataset)
#
# # evaluate model on test set
# test_epoch = smp.utils.train.ValidEpoch(
#     model=best_model,
#     loss=loss,
#     metrics=metrics,
#     device=DEVICE,
# )
#
# logs = test_epoch.run(test_dataloader)
#
# # test dataset without transformations for image visualization
# test_dataset_vis = Dataset(
#     x_test_dir, y_test_dir,
#     classes=CLASSES,
# )
#
# for i in range(3):
#     # n = np.random.choice(len(test_dataset))
#
#     image_vis = test_dataset_vis[i][0].astype('uint8')
#     image, gt_mask = test_dataset[i]
#
#     gt_mask = gt_mask.squeeze()
#
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     pr_mask = best_model.predict(x_tensor)
#     pr_mask = (pr_mask.squeeze().cpu().numpy().round())
#
#     visualize(
#         image=image_vis,
#         ground_truth_mask=gt_mask[0, ...].squeeze(),
#         animal=pr_mask[0, ...].squeeze(),
#         masking=pr_mask[1, ...].squeeze(),
#         nonmasking=pr_mask[2, ...].squeeze(),
#         attention=pr_mask[3, ...].squeeze(),
#     )
import os

import cv2
import numpy as np

import config
import utils
from torch.utils.data import Dataset as BaseDataset

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
        self.class_values = [config.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = utils.crop_to_divisible_by_32(image)
        mask = utils.crop_to_divisible_by_32(mask)

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
            color = config.COLORS[cls_value]
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

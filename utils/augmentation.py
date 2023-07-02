import albumentations as albu


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.9, border_mode=0),
        albu.PadIfNeeded(min_height=544, min_width=544, always_apply=True, border_mode=0),
        albu.RandomCrop(height=544, width=544, always_apply=True),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.5),
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                albu.RandomGamma(gamma_limit=(80, 120), p=0.5),
                albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            ],
            p=0.8,
        ),
        albu.OneOf(
            [
                albu.GaussianBlur(blur_limit=(3, 7), p=0.5),
                albu.MedianBlur(blur_limit=3, p=0.5),
                albu.MotionBlur(blur_limit=(3, 7), p=0.5),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                albu.ImageCompression(quality_lower=85, quality_upper=95, p=0.5),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(544, 544, always_apply=True),
        #albu.RandomCrop(height=512, width=512, always_apply=True),
    ]
    return albu.Compose(test_transform)

def get_test_augmentation():
    test_transform = [
        albu.PadIfNeeded(544, 544, always_apply=True)
        #albu.Normalize(),
    ]
    return albu.Compose(test_transform)

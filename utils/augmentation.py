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
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=(3, 7), p=1),
                albu.MotionBlur(blur_limit=(3, 7), p=1),
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

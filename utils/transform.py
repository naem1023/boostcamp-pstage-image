import albumentations as A
import albumentations.pytorch

transformation = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.OneOf([A.GaussNoise()], p=0.2),
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.HueSaturationValue(),
                A.RGBShift(),
                A.ChannelShuffle(),
            ],
            p=0.3,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=10,
            border_mode=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=(-30, 30), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)

import albumentations as A
from albumentations.pytorch import ToTensorV2
from copy_paste import CopyPaste

class BaseAugmentation:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
            ])

class BaseCopyPasteAugmentation:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), data_root = "../input/data", json_dir = "./splited_json/train_split_0.json"):
        self.transform = A.Compose([
            CopyPaste(data_root = data_root, json_dir = json_dir, p = 1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


class NoAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            ToTensorV2()
        ])


class JDJ_Augmentation:
    def __init__(self, data_root, json_dir):
        self.transform = A.Compose([
            CopyPaste(data_root=data_root, json_dir=json_dir, p=1),
            ##################
            # shape          #
            ##################
            A.Rotate(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OpticalDistortion(),

            ##################
            # color or noise #
            ##################
            A.ColorJitter(),
            A.Sharpen(p=0.2),
            A.ISONoise(),
            A.GlassBlur(),
            A.OneOf([
                A.RandomShadow(),
                # 시간대에 따른 그림자
                A.RandomSnow()
                # 겨울철 눈 배경 사진
            ], p=0.25),

            A.OneOf([
                A.RandomBrightnessContrast(),
                A.RandomToneCurve()
            ], p=0.25),
            # 밤낮에 따른 배경 밝기

            ##################
            # delete         #
            ##################
            A.OneOf([
                A.GridDropout(),
                A.CoarseDropout()
            ], p=0.25),

            ToTensorV2()
        ])


class JYAugmentation:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), data_root="../input/data", json_dir="./splited_json/train_split_0.json"):
        self.transform = A.Compose([
            CopyPaste(data_root = data_root, json_dir = json_dir, p = 1),
            A.RandomResizedCrop(height = 512, width = 512, scale = (0.6, 1)),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.2),
            A.RandomRotate90(p = 0.5),

            A.OneOf([
                A.ElasticTransform(p = 1, alpha = 240, sigma = 240*0.05, alpha_affine = 240 * 0.03),
                A.GridDistortion(p = 1),
                A.OpticalDistortion(distort_limit = 1, shift_limit = 0.1, p = 1)
            ], p = 0.6),

            A.OneOf([
                A.HueSaturationValue(hue_shift_limit = 0.2,
                                     sat_shift_limit = 0.2,
                                     val_shift_limit = 0.2,
                                     p = 0.2),
                A.RandomBrightnessContrast(brightness_limit = 0.2,
                                           contrast_limit = 0.1,
                                           p = 0.2)
            ], p = 0.2),

            A.OneOf([
                A.Blur(p = 0.5),
                A.GaussianBlur(p = 0.5),
                A.Sharpen(p = 0.5)
            ], p = 0.3),

            A.GridDropout(ratio = 0.25, p = 0.3),
            A.Normalize(mean = mean, std = std),
            ToTensorV2()
        ])

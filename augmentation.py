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
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = A.Compose([
            CopyPaste(data_root = "../input/data", json_dir = "./splited_json/train_split_0.json", p = 1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
            ])
            
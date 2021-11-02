from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torch
import albumentations as A
import numpy as np
import random
import cv2
import os

class TrainDataset(Dataset):

    category_names = ['Backgroud','General trash','Paper','Paper pack','Metal','Glass',
                      'Plastic','Styrofoam','Plastic bag','Battery','Clothing']

    def __init__(self, data_root, json_dir, binary_class = "Metal", mode = "train", cutmix_prob = 0.25, mixup_prob = 0.25, transform = None):

        """ Trash Object Detection Train Dataset
        Args:
            data_root : root for data
            json_dir : directory for annotation json file
            mode : "train" when you want to train, "validation" when you want to evaluate
            cutmix_prob : probability of applying a cutmix
            mixup_prob : probability of applying a mixup
            transform : transform to be applied to the image
        """
        
        super().__init__()
        self.data_root = data_root
        self.coco = COCO(json_dir)
        self.mode = mode
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
        self.transform = transform
        self.num_classes = len(self.category_names)
        self.num_images = len(self.coco.imgs)
        self.binary_class = binary_class
        
        self.img_idx = []
        for coco_img in self.coco.imgs:
            self.img_idx.append(coco_img)


    def __len__(self):

        return self.num_images


    def __getitem__(self, index):
        
        random_number = random.random()
        
        if self.mode == "validation":
            image, mask = self.load_image_mask(index)

        elif self.mode == "train":
            if random_number > 1-self.mixup_prob:
                image, mask = self.load_mixup(index)
            elif random_number > 1 - self.mixup_prob - self.cutmix_prob:
                image, mask = self.load_cutmix(index)
                if self.mixup_prob > 0:
                    mask = self.mask_to_prob(mask)
            else:
                image, mask = self.load_image_mask(index)
                if self.mixup_prob > 0:
                    mask = self.mask_to_prob(mask)

        if self.transform:
            transformed = self.transform(image = image, mask = mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
        
        return image.float(), mask

    
    def load_image_mask(self, index):
        image_id = self.coco.getImgIds(imgIds = self.img_idx[index])
        image_info = self.coco.loadImgs(image_id)[0]

        image = cv2.imread(os.path.join(self.data_root, image_info['file_name']))   # uint8
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)
        
        mask = np.zeros((image_info["height"], image_info["width"]))
        
        anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=True)
        for i in range(len(anns)):
            className = self.get_classname(anns[i]['category_id'], cats)
            if className == self.binary_class:
                mask[self.coco.annToMask(anns[i]) == 1] = 1
        mask = mask.astype(np.int8)

        return image, mask

    def load_cutmix(self, index, img_size = 512):

        w, h = img_size, img_size
        s = img_size // 2

        xc, yc = [int(random.uniform(img_size * 0.25, img_size * 0.75)) for _ in range(2)] 
        indexes = [index] + [random.randint(0, self.num_images - 1) for _ in range(3)]

        result_image = np.full((img_size, img_size, 3), 1, dtype = np.uint8)
        result_mask = np.full((img_size, img_size), 0, dtype = np.int8)

        for i, index in enumerate(indexes):
            image, mask = self.load_image_mask(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                
            elif i == 1:  
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                
            elif i == 2: 
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                
            elif i == 3: 
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                
            transformed = A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(y2a - y1a, x2a - x1a)(image = image, mask = mask)
            result_image[y1a:y2a, x1a:x2a] = transformed['image']
            result_mask[y1a:y2a, x1a:x2a] = transformed['mask']

        return result_image, result_mask


    def load_mixup(self, index):

        random_number = random.uniform(0.4, 0.6)

        indexes = [index, random.randint(0, self.num_images - 1)]

        image1, mask1 = self.load_image_mask(indexes[0])
        image2, mask2 = self.load_image_mask(indexes[1])

        result_image = image1 * random_number + image2 * (1 - random_number)

        mask1 = self.mask_to_prob(mask1)
        mask2 = self.mask_to_prob(mask2)
        
        result_mask = mask1 * random_number + mask2 * (1 - random_number)

        return result_image, result_mask


    def _get_area_cat(self, seg):

        linspace = np.linspace(0, 5000, 6)
        s = 0

        if type(seg) != list:
            s = 0
        else:
            for _seg in seg:
                s += len(_seg)

        if s < linspace[1]:
                return 0
        elif s < linspace[2]:
            return 1
        elif s < linspace[3]:
            return 2
        elif s < linspace[4]:
            return 3
        elif s < linspace[5]:
            return 4
        else:
            return 5


    def get_classname(self, classID, cats):

        for i in range(len(cats)):
            if cats[i]['id']==classID:

                return cats[i]['name']

        return "None"


    def mask_to_prob(self, mask, img_size = 512):

        lst = []
        for i in range(img_size):
            lst.append(np.eye(11)[mask[i]])
        mask = np.transpose(np.stack(lst), (2, 0, 1))
        return mask


class TestDataset(Dataset):

    def __init__(self, data_root, json_dir, transform = None):

        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.coco = COCO(json_dir)
        self.num_images = len(self.coco.imgs)
        self.img_idx = []
        for coco_img in self.coco.imgs:
            self.img_idx.append(coco_img)

        
    def __getitem__(self, index: int):

        image_id = self.coco.getImgIds(imgIds=self.img_idx[index])
        image_infos = self.coco.loadImgs(image_id)[0]
        
        images = cv2.imread(os.path.join(self.data_root, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        if self.transform is not None:
            transformed = self.transform(image=images)
            images = transformed["image"]
        return images, image_infos
    

    def __len__(self):
   
        return self.num_images
        

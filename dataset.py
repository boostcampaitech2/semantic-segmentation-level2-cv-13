from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import random
import torch
import json
import cv2
import os


class TrainDataset(Dataset):

    category_names = ['Backgroud','General trash','Paper','Paper pack','Metal','Glass',
                      'Plastic','Styrofoam','Plastic bag','Battery','Clothing']

    def __init__(self, annotation, data_dir, mode, fold = 0, k = 5, cutmix_prob = 0.25, mixup_prob = 0.25, random_state = 923, transform = None):
        
        """ Trash Object Detection Train Dataset
        Args:
            annotation : annotation directory
            data_dir : data_dir directory
            mode : "train" when you want to train, "validation" when you want to evaluate
            cutmix_prob : probability of applying a cutmix
            mixup_prob : probability of applying a mixup
            fold : the order of fold to be learned
            k : how many folds going to devided
            random_state : random state of kfold
            transform : transform to be applied to the image
        """
        
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.mode = mode
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
        self.transform = transform
        self.num_classes = len(self.category_names)

        with open(annotation) as f:
            train = json.load(f)

        self.coco = COCO(annotation)
        df_images = pd.json_normalize(train['images'])
        df_annotations = pd.json_normalize(train['annotations'])
        train_df = df_images.set_index('id').join(df_annotations.set_index('image_id'))
        train_df['image_id'] = train_df.index
        train_df['seg_area_cat'] = train_df['segmentation'].apply(self._get_area_cat)
        train_df['cat'] = train_df['category_id'].astype(str) + "_" + train_df['seg_area_cat'].astype(str)
        
        skf = StratifiedGroupKFold(n_splits = k, random_state = 923, shuffle = True)
        for i, (train_idx, val_idx) in enumerate(skf.split(train_df['id'], train_df['cat'], train_df['image_id'])):
            if i == fold:
                break

        if self.mode == "train":
            self.img_idx = train_df.iloc[train_idx]['image_id'].unique()
        if self.mode == "validation":
            self.img_idx = train_df.iloc[val_idx]['image_id'].unique()


    def __len__(self):

        return len(self.img_idx)


    def __getitem__(self, index):
        
        random_number = random.random()
        
        if self.mode == "validation":
            image, mask, image_info = self.load_image_mask(index)

        elif self.mode == "train":
            if random_number > 1-self.mixup_prob:
                image, mask, image_info = self.load_mixup(index)
            elif random_number > 1 - self.mixup_prob - self.cutmix_prob:
                image, mask, image_info = self.load_cutmix(index)
            else:
                image, mask, image_info = self.load_image_mask(index)

            if self.mixup_prob > 0:
                mask = self.mask_to_prob(mask)

        if self.transform:
            transformed = self.transform(image = image, mask = mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask, image_info

    
    def load_image_mask(self, index):
        image_id = self.coco.getImgIds(imgIds = self.img_idx[index])
        image_info = self.coco.loadImgs(image_id)[0]

        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)

        
        mask = np.zeros((image_info["height"], image_info["width"]))
        
        anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
        for i in range(len(anns)):
            className = self.get_classname(anns[i]['category_id'], cats)
            pixel_value = self.category_names.index(className)
            mask[self.coco.annToMask(anns[i]) == 1] = pixel_value
        mask = mask.astype(np.int8)        

        return image, mask, image_info

    def load_cutmix(self, index, img_size = 512):

        w, h = img_size, img_size
        s = img_size // 2

        image_id = self.coco.getImgIds(imgIds = self.img_idx[index])
        image_info = self.coco.loadImgs(image_id)[0]

        xc, yc = [int(random.uniform(img_size * 0.25, img_size * 0.75)) for _ in range(2)] 
        indexes = [index] + [random.randint(0, len(self.img_idx) - 1) for _ in range(3)]

        result_image = np.full((img_size, img_size, 3), 1, dtype = np.float32)
        result_mask = np.full((img_size, img_size), 0, dtype = np.int8)

        for i, index in enumerate(indexes):
            image, mask, _ = self.load_image_mask(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2: 
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3: 
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            result_mask[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]

        return result_image, result_mask, image_info


    def load_mixup(self, index, img_size = 512):

        image_id = self.coco.getImgIds(imgIds = self.img_idx[index])
        image_info = self.coco.loadImgs(image_id)[0]

        indexes = [index, random.randint(0, len(self.img_idx) - 1)]

        image1, mask1, image_info1 = self.load_image_mask(indexes[0])
        image2, mask2, image_info2 = self.load_image_mask(indexes[1])

        result_image = (image1 + image2) / 2

        mask1 = self.mask_to_prob(mask1)
        mask2 = self.mask_to_prob(mask2)
        
        result_mask = (mask1 + mask2) / 2

        return result_image, result_mask, image_info


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
        mask = np.stack(lst)

        return mask


class TestDataset(Dataset):

    def __init__(self, annotation, data_dir, transform = None):

        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.coco = COCO(annotation)

        
    def __getitem__(self, index: int):

        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        images = cv2.imread(os.path.join(self.data_dir, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if self.transform is not None:
            transformed = self.transform(image=images)
            images = transformed["image"]
        return images, image_infos
    

    def __len__(self):
   
        return len(self.coco.getImgIds())
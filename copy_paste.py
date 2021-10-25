from pycocotools.coco import COCO
import json
import pandas as pd
import albumentations as A
import random
import cv2

class CopyPaste(A.DualTransform):
    def __init__(self, data_root = "../input/data", json_dir = "./splited_json/train_split_0.json", p = 0.5, always_apply = False):
        super(CopyPaste, self).__init__(always_apply, p)

        self.coco = COCO(json_dir)
        with open(json_dir) as f:
            train = json.load(f)

        self.data_root = data_root
        df_images = pd.json_normalize(train['images'])
        df_annotations = pd.json_normalize(train['annotations'])
        self.train_df = pd.merge(df_annotations, df_images, how = 'left', left_on = 'image_id', right_on = 'id')

        self.idx1 = self.train_df.query("category_id == 1").index
        self.idx2 = self.train_df.query("category_id == 2").index
        self.idx3 = self.train_df.query("category_id == 3").index
        self.idx4 = self.train_df.query("category_id == 4").index
        self.idx5 = self.train_df.query("category_id == 5").index
        self.idx6 = self.train_df.query("category_id == 6").index
        self.idx7 = self.train_df.query("category_id == 7").index
        self.idx8 = self.train_df.query("category_id == 8").index
        self.idx9 = self.train_df.query("category_id == 9").index
        self.idx10 = self.train_df.query("category_id == 10").index

        self._p = [0, 0.02494392, 0.11797518, 0.21561238, 0.33476895, 0.46415433, 0.59427247, 0.72510842, 0.86170181, 1]
        self.idx = None
        self.ann = None

    def apply(self, img, **params):
        self.choice_index()
        img[self.seg == 1] = self.new_img[self.seg == 1]
        return img

    def apply_to_mask(self, img, **params):
        img[self.seg == 1] = self.ann['category_id']
        return img
    
    def choice_index(self):
        randn = random.random()
        if randn > self._p[8]:
            idx = random.choice(self.idx9)
        elif randn > self._p[7]:
            idx = random.choice(self.idx10)
        elif randn > self._p[6]:
            idx = random.choice(self.idx4)
        elif randn > self._p[5]:
            idx = random.choice(self.idx5)
        elif randn > self._p[4]:
            idx = random.choice(self.idx3)
        elif randn > self._p[3]:
            idx = random.choice(self.idx7)
        elif randn > self._p[2]:
            idx = random.choice(self.idx1)
        elif randn > self._p[1]:
            idx = random.choice(self.idx6)
        else:
            idx = random.choice(self.idx8)
        self.idx = idx
        self.ann = {'id' : 0, 'image_id' : 0, 'category_id' : self.train_df.iloc[self.idx]['category_id'],
                    'segmentation' : self.train_df.iloc[self.idx]['segmentation']}
        self.fn = self.train_df.iloc[self.idx]['file_name']
        self.new_img = cv2.imread(self.data_root + "/" + self.fn)
        self.seg = self.coco.annToMask(self.ann)
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

        self.indexes = [self.train_df.query("category_id == 8").index,
                        self.train_df.query("category_id == 6").index,
                        self.train_df.query("category_id == 1").index,
                        self.train_df.query("category_id == 7").index,
                        self.train_df.query("category_id == 3").index,
                        self.train_df.query("category_id == 5").index,
                        self.train_df.query("category_id == 4").index,
                        self.train_df.query("category_id == 10").index,
                        self.train_df.query("category_id == 9").index]

        self._p = [0., 0.00511753, 0.07630264, 0.15471094, 0.27149064, 0.40918043, 0.54843422, 0.68922868, 0.8426871 , 1.]
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
        randn = random.random() # [0, 1)

        for i in range(9):
            if self._p[i] < randn < self._p[i + 1]:
                break
        
        self.idx = random.choice(self.indexes[i])
        self.ann = {'id' : 0, 'image_id' : 0, 'category_id' : self.train_df.iloc[self.idx]['category_id'],
                    'segmentation' : self.train_df.iloc[self.idx]['segmentation']}
        self.fn = self.train_df.iloc[self.idx]['file_name']
        self.new_img = cv2.imread(self.data_root + "/" + self.fn)
        self.seg = self.coco.annToMask(self.ann)

    def get_transform_init_args_names(self):
        return ()
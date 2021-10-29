from pycocotools.coco import COCO
import json
import pandas as pd
import albumentations as A
import random
import cv2
import copy
import math
import numpy as np

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


class CopyPasteV2(A.DualTransform):
    '''
    기존 CopyPaste의 수정된 버전이다.
    Copy-Paste 대상이 되는 이미지와 마스크를 정제하고 싶을때(ex. 특정 크기 이상),
    미리 해당 정보만을 담은 json파일을 만들고 이 class를 사용하는 것을 추천한다.
    이와 관련한 json을 만드는 것은 아래 참고
    https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-13/blob/main/etc/split_battery_except_small.ipynb

    기존과의 차이점은
    1. 기존에서는 클래별 추출 확률값을 줬지만, 여기서는 클래스별 추출 확률값을 주지 않는다.
       클래스별 추출 확률값을 고려하지 않아도 되는 경우에 해당 클래스를 쓰면 된다.
    2. Copy-Paste 대상이 되는 이미지와 마스크를 random하게 resize를 한다.
    3. Copy-Paste 대상이 되는 이미지와 마스크를 random하게 위치를 조정시킨다.
    4. Copy-Paste 대상이 되는 이미지와 마스크를 random하게 회전시킨다.

    Args:
        data_root (str): 이미지파일의 상위폴더 경로 (변경해줄 필요 없음)
        json_dir (str): Copy-Paste의 대상이 되는 json 파일
        p (float): Copy-Paste가 적용되는 확률값
        min_resize (int): Copy-Paste 대상이 되는 이미지와 마스크를 random하게 resize를 할때,
                          bbox 최소 면적이 min_resize*min_resize되게 한다.
        max_resize (int): Copy-Paste 대상이 되는 이미지와 마스크를 random하게 resize를 할때,
                          bbox 최대 면적이 max_resize*max_resize되게 한다.
                          만약 대상이 되는 bbox의 면적이 max_resize*max_resize가 넘지 않는 경우,
                          resize는 수행되지 않는다.
        min_rotate (float): Copy-Paste 대상이 되는 이미지와 마스크를 random하게 회전시킬 때,
                            최소 각도
        max_rotate (float): Copy-Paste 대상이 되는 이미지와 마스크를 random하게 회전시킬 때,
                            최대 각도 
        always_apply (bool): 해당 augmentation의 적용 확률값이 항상 1이 되게 할지 설정
    '''
    def __init__(self, data_root="../input/data", json_dir="/opt/ml/segmentation/input/data/category_split_json/clothing_200_200.json", \
        p=0.5, min_resize=100, max_resize=150, min_rotate=-30, max_rotate=30, always_apply=False):
        super(CopyPasteV2, self).__init__(always_apply, p)

        self.coco = COCO(json_dir)
        self.data_root = data_root
        with open(json_dir, 'r') as f:
            json_file = json.load(f)
        self.annotations = json_file['annotations']
        self.min_resize = min_resize
        self.max_resize = max_resize
        self.min_rotate = min_rotate
        self.max_rotate = max_rotate

        self.transform = A.Compose([
            A.Rotate(limit=(self.min_rotate, self.max_rotate))
        ])

        self.pad_length = 50

    def apply(self, img, **params):
        self.choice_index()
        img[self.seg != 0] = self.new_img[self.seg != 0] # mask가 있는 부분
        return img

    def apply_to_mask(self, img, **params):
        img[self.seg != 0] = self.seg[self.seg != 0]
        return img
    
    def choice_index(self):
        self.ann = random.choice(self.annotations)
        random_resize = random.randint(self.min_resize, self.max_resize)
        
        img_idx = self.ann['image_id']
        new_img = cv2.imread(self.data_root + "/" + self.coco.loadImgs(img_idx)[0]['file_name'])
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        seg = np.zeros((new_img.shape[0], new_img.shape[1]))
        
        category_id = self.ann['category_id']
        seg[self.coco.annToMask(self.ann) == 1] = category_id

        # target 이미지 ann bbox 위치 random하게 재조정
        x = math.floor(self.ann['bbox'][0])
        y = math.floor(self.ann['bbox'][1])
        w = math.ceil(self.ann['bbox'][2])
        h = math.ceil(self.ann['bbox'][3])

        new_w = None
        new_h = None

        # target 이미지 ann bbox 크기 random하게 재조정
        if w * h > self.max_resize * self.max_resize:
            new_w = math.ceil(w * (random_resize / math.sqrt(w*h)))
            new_h = math.ceil(h * (random_resize / math.sqrt(w*h)))

        patch = copy.deepcopy(new_img[y:y+h, x:x+w])
        zero_patch = np.zeros_like(patch)
        patch = cv2.resize(patch, dsize=(new_w, new_h))

        # 가운데 부분에는 paste가 되지 않도록 위치 조정
        rand_int = random.randint(1, 4)
        if rand_int == 1:
            new_x = random.randint(0, self.pad_length)
            new_y = random.randint(0, self.pad_length)
        elif rand_int == 2:
            new_x = random.randint(0, self.pad_length)
            new_y = random.randint(512-new_h-self.pad_length, 512-new_h)
        elif rand_int == 3:
            new_x = random.randint(512-new_w-self.pad_length, 512-new_w)
            new_y = random.randint(0, self.pad_length)
        else:
            new_x = random.randint(512-new_w-self.pad_length, 512-new_w)
            new_y = random.randint(512-new_h-self.pad_length, 512-new_h)

        # target이미지의 bounding box 부분이 새로운 위치로 조정
        new_img[y:y+h, x:x+w] = zero_patch 
        new_img[new_y:new_y+new_h, new_x:new_x+new_w] = patch

        patch_mask = copy.deepcopy(seg[y:y+h, x:x+w])
        zero_patch_mask = np.zeros_like(patch_mask)
        patch_mask = cv2.resize(patch_mask, dsize=(new_w, new_h))
        
        # target이미지의 segmentation 위치가 새로운 위치로 조정
        seg[y:y+h, x:x+w] = zero_patch_mask
        seg[new_y:new_y+new_h, new_x:new_x+new_w] = patch_mask

        transformed = self.transform(image=new_img, mask=seg)
        new_img = transformed['image']
        seg = transformed['mask']

        self.new_img = new_img
        self.seg = seg

    def get_transform_init_args_names(self):
        return ()
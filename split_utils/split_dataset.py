from splitutils import df_to_formatted_json
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from collections import Counter, defaultdict
from pycocotools.coco import COCO
import json
import pandas as pd
import numpy as np
import argparse 
import time
import os

def get_distribution(y_vals):
        y_distr = Counter(y_vals)
        y_vals_sum = sum(y_distr.values())
        return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)] + [y_vals_sum]


def _get_area_cat(seg):
        linspace = [0,150, 400, 1100, 2200] #np.linspace(0, 5000, 6)
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
        # elif s < linspace[5]:
        #     return 4
        else:
            return 4


if __name__=="__main__":
    start_time = time.time()

    # parser.add_argument('--split_fold', type = int, help='validation split fold',default=0)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--json_path', type = str, 
                        help='json file path',default='/opt/ml/segmentation/input/data/train_all.json')
    parser.add_argument('--split_num', type = str, 
                        help='fold split number',default=5)
    parser.add_argument('--seed', type = str, 
                        help='fold split seed',default=42)
    parser.add_argument('--json_save_folder', type = str, 
                        help='folder_directory to save new json file', default='/opt/ml/segmentation/semantic-segmentation-level2-cv-13/splited_json/')
    # parser.add_argument('--check_outlier',type =bool,  default = False, help = "check whether remove outlier or not")
    # parser.add_argument('--rm_bbox',type =int,  default = 40, 
    #     help = "the number of annotations that a image is defined as outlier")
    # parser.add_argument('--rm_wh', type = list, default = [10,10],
    #     help = "Width and Height of annotations to be defined as outlier")
    args = parser.parse_args() 

    
    # train 파일을 json과 coco type으로 불러오기
    with open(args.json_path) as f:
            train = json.load(f)

    coco = COCO(args.json_path)

    # image category와 instance별 넓이의 분포를 맞춰 split 하는 부분
    df_images = pd.json_normalize(train['images'])
    df_annotations = pd.json_normalize(train['annotations'])
    train_df = df_annotations.merge(df_images.rename(columns ={'image_id':'id'}), how = 'left', on = 'id') 
    
    train_df['seg_area_cat'] = train_df['segmentation'].apply(_get_area_cat)
    train_df['cat'] = train_df['category_id'].astype(str) + "_" + train_df['seg_area_cat'].astype(str)
    
    skf = StratifiedGroupKFold(n_splits = args.split_num, random_state = args.seed, shuffle = True)
    folds = skf.split(train_df['id'], train_df['cat'], train_df['image_id'])

    # 출력 확인을 위한 부분
    distrs = [get_distribution(train_df['category_id'])]
    index = ['training set']

    for fold, (trn_idx, val_idx) in enumerate(folds):

        print(f"{fold} spliting...")
        # 원본 train json 이용하여 image, annotations 바꿔준다. 
        trn_json = train.copy()
        val_json = train.copy()

        trn_json['images'] = df_to_formatted_json(df_images.loc[df_images['id'].isin(df_annotations.loc[trn_idx,'image_id'].unique())]) 
        val_json['images'] = df_to_formatted_json(df_images.loc[df_images['id'].isin(df_annotations.loc[val_idx,'image_id'].unique())])

        trn_json['annotations'] = df_to_formatted_json(df_annotations.loc[trn_idx])
        val_json['annotations'] = df_to_formatted_json(df_annotations.loc[val_idx])


        # image id와 annotation id를 연속적이게 바꿔주는 부분
        # train
        start_ann_idx = 0
        trn_img_df = pd.json_normalize(trn_json['images'])
        for i, img_info in enumerate(trn_json['images']):
            image_id = trn_img_df['id'].iloc[i]
            ann_ids = coco.getAnnIds(imgIds=image_id)

            for ann_idx in range(start_ann_idx, start_ann_idx+len(ann_ids)):
                trn_json['annotations'][ann_idx]['image_id'] = i # annotation의 image_id 바꾼다.
                trn_json['annotations'][ann_idx]['id'] = ann_idx # annotations id 바꾼다.
            start_ann_idx = ann_idx + 1
            trn_json['images'][i]['id'] = i # image의 id 바꾼다.

        # valid
        start_ann_idx = 0
        val_img_df = pd.json_normalize(val_json['images'])
        for i, img_info in enumerate(val_json['images']):
            image_id = val_img_df['id'].iloc[i] # # 이미지 불러와서  
            ann_ids = coco.getAnnIds(imgIds=image_id) # 어노테이션 아디들 불러오고

            for ann_idx in range(start_ann_idx, start_ann_idx+len(ann_ids)):
                val_json['annotations'][ann_idx]['image_id'] = i
                val_json['annotations'][ann_idx]['id'] = ann_idx 
            start_ann_idx = ann_idx + 1
            val_json['images'][i]['id'] = i

        # folder 생성
        try: 
            if not os.path.exists(args.json_save_folder): 
                os.makedirs(args.json_save_folder) 
        except OSError: 
            print("Error: Failed to create the directory.")

        # if args.check_outlier:
        #     trn_json = rm_outlier(trn_json, args.rm_bbox, args.rm_wh)
        # #    val_json = rm_outlier(val_json, args.rm_bbox, args.rm_wh)

        with open(args.json_save_folder + f'/train_split_{fold}.json', 'w') as fp:
            json.dump(trn_json, fp)

        with open(args.json_save_folder + f'/valid_split_{fold}.json', 'w') as fp:
            json.dump(val_json, fp)
        
        # 잘 나뉘는지 결과 검증을 위한 부분
        dev_groups, val_groups = train_df['image_id'][trn_idx], train_df['image_id'][val_idx]    
        assert len(set(dev_groups) & set(val_groups)) == 0
        
        distrs.append(get_distribution(train_df['category_id'][trn_idx]))
        index.append(f'development set - fold {fold}')
        distrs.append(get_distribution(train_df['category_id'][val_idx]))
        index.append(f'validation set - fold {fold}')

        print(f"{fold} end!")
    
    print(f"json files saved at {args.json_save_folder}")
    print(f"---{np.round(time.time()-start_time, 2)}s seconds---")
    print(pd.DataFrame(distrs, index=index, columns=[f'Label {l}' for l in range(np.max(train_df['category_id']) + 1)] + ['Total_num']))

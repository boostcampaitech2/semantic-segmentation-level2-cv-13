import pandas as pd

def df_to_formatted_json(df, sep="."):
    """
    The opposite of json_normalize, make pandas dataframe into json 
    pandas dataframe을 json 파일로 만들어 주는 함수입니다.
    """
    result = []
    for idx, row in df.iterrows():
        parsed_row = {}
        for col_label,v in row.items():
            keys = col_label.split(".")

            current = parsed_row
            for i, k in enumerate(keys):
                if i==len(keys)-1:
                    current[k] = v
                else:
                    if k not in current.keys():
                        current[k] = {}
                    current = current[k]
        # save
        result.append(parsed_row)
    return result

def rm_outlier(target_json, bbox_num = 40, wh = [10,10]):
#         with open(args.json_file) as f:
#         train = json.load(f)
    
    df_images = pd.json_normalize(target_json['images']) # 이미지 정보
    df_annotations = pd.json_normalize(target_json['annotations']) #  annotation 정보
    origin_annotation_columns = df_annotations.columns # 나중에 쓰여요
    origin_img_columns = df_images.columns # 나중에 쓰여요
    train_df = df_annotations.merge(df_images.rename(columns = {'id':'image_id'}), how = 'left',on = 'image_id') # 이미지 정보와 annotation 정보 합치기

    train_df['width'] = train_df['bbox'].apply(lambda x: x[2])
    train_df['height'] = train_df['bbox'].apply(lambda x: x[3])
    print(f"원본 파일의 image 수: {train_df['image_id'].nunique():,}, annotation 수: {len(train_df):,}")

    # bounding box 수로 제거하는 부분
    ann_cnt = train_df['image_id'].value_counts()
    outlier_idx = ann_cnt.loc[ann_cnt > bbox_num].index # 이미지에 bbox수가 지정 개수보다 많으면 아웃라이어로 지정

    df_images_rm = df_images.loc[~df_images['id'].isin(outlier_idx)].reset_index(drop=True).copy() # 현제 image정보에서는 index와 image id 동일하기에 해당 index row drop
    id_maps = {k: v for k, v, in zip(df_images_rm['id'].tolist(), list(range(0,df_images_rm.shape[0])))} # 원래 image_id가 무었이었는지 정보 저장
    df_images_rm['id'] = df_images_rm.index # id 연속적으로 바꿔주기 
    
    # annotation에서 삭제된 이미지에 포함된 모든 annotation 삭제
    train_df_rm = train_df.loc[~train_df['image_id'].isin(outlier_idx)].reset_index(drop=True).copy()

    # annoation width, height로 제거하는 부분
    train_df_rm['image_id'] = train_df_rm['image_id'].map(id_maps) # 새로운 image id로 매핑
    train_df_rm = train_df_rm.loc[(train_df_rm['width']>=wh[0]) | (train_df_rm['height']>=wh[1])].reset_index(drop=True).copy() # 설정한 width height
    
    # 작은 박스들을 지우고 난 뒤에 혹시 annotation이 하나도 없는 이미지가 있을수도 있다 만약의 경우!
    # 이를 확인하는 방법은 train_df_rm에 image_id의 개수가 기존보다 적어졌다면 그런 문제가 발생했을것(image_id를 mapping 해주었기 때문에)
    # 근데 이런 경우 거의 없을것
    if df_images_rm['id'].nunique() > train_df_rm['id'].nunique():
        print("outlier가 너무 많이 제거 됬을 수도 있습니다.")
        df_images_rm = df_images_rm.loc[df_images_rm['id'].isin(train_df_rm['image_id'].unique())].drop_index(drop=True).copy()
        id_maps = {k: v for k, v, in zip(df_images_rm['id'].tolist(), list(range(0,df_images_rm.shape[0])))}
        df_images_rm['id'] = df_images_rm.index
        train_df_rm['image_id'] = train_df_rm['image_id'].map(id_maps)

    train_df_rm['id'] = train_df_rm.index
    df_annotation_rm = train_df_rm[origin_annotation_columns] 
    df_images_rm = df_images_rm[origin_img_columns]

    print(f"이상치 제거 파일의 image 수: {train_df_rm['image_id'].nunique():,}, annotation 수: {len(train_df_rm):,}")

    target_json['images'] = df_to_formatted_json(df_images_rm)
    target_json['annotations'] = df_to_formatted_json(df_annotation_rm)
    
    return target_json
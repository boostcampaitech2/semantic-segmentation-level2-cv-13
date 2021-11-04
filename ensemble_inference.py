import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import ttach as tta
from tqdm import tqdm
import os
from importlib import import_module

from utils import fix_seed, arg_parse
import json
from collections import namedtuple

import albumentations as A
from multiprocessing import Pool

import time


def collate_fn(batch):
    return tuple(zip(*batch))


def albu_transform(oms):
    tmp_mask = []
    tf = A.Compose([A.Resize(256, 256)])
    for mask in oms:
        tmp_mask.append(tf(image=np.zeros_like(mask, dtype=np.uint8), mask=mask)[
                        'mask'].reshape(1, 256*256).astype(int))

    return np.array(tmp_mask)


if __name__ == "__main__":
    start = time.time()
    args = arg_parse()

    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple(
            'x', d.keys())(*d.values()))

    # seed 고정
    fix_seed(cfgs.seed)

    size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_name = "ensemble_results"
    for model_info in cfgs.model_infos:
        output_name += "_" + model_info.name
    

    # weight = torch.tensor(cfgs.weight, dtype=torch.float32, device=device)
    weight = cfgs.weight
    test_dataset_module = getattr(import_module("dataset"), cfgs.dataset)

    total_result_list = []    # (num_batches, batch_size, class_num, w, h)
    file_names = []

    for model_idx, model_info in enumerate(cfgs.model_infos):
        print(f"Model [{model_idx+1}] START")
        print(model_info)
        test_augmentation_module = getattr(
            import_module("augmentation"), model_info.augmentation)
        test_augmentation = test_augmentation_module().transform

        test_dataset = test_dataset_module(data_root=cfgs.data_root, json_dir=cfgs.json_path,
                                           transform=test_augmentation)
        test_loader = DataLoader(dataset=test_dataset,
                                 **cfgs.dataloader.args._asdict(),
                                 collate_fn=collate_fn)

        # make file_names once
        if not file_names:
            file_name_list = []
            for _, image_infos in test_loader:
                file_name_list.append([i['file_name'] for i in image_infos])
            file_names = [y for x in file_name_list for y in x]

        # model arch
        model_module = getattr(import_module("model"), model_info.name)
        model = model_module(**model_info.args._asdict()).to(device)

        # load model weights
        checkpoint = torch.load(model_info.weight_path, map_location=device)
        state_dict = checkpoint['net']

        model.load_state_dict(state_dict)

        if model_info.tta:
            tta_transform = getattr(import_module(
                "ttach.aliases"), model_info.tta.name)
            model = tta.SegmentationTTAWrapper(model, tta_transform(
                **model_info.tta.args._asdict()), output_mask_key='out')

        model_result_list = []
        with torch.no_grad():
            for imgs, image_infos in tqdm(test_loader):
                # inference (512 x 512)
                imgs = torch.stack(imgs).to(device)
                outs = model(imgs)['out']
                oms = outs.detach().cpu().numpy()

                model_result_list.append(oms)

        if total_result_list:  # not empty
            for list_idx, result in enumerate(model_result_list):
                total_result_list[list_idx] += weight[model_idx] * result
        else:
            for list_idx, result in enumerate(model_result_list):
                total_result_list.append(weight[model_idx] * result)

    # 배치별 argmax 진행
    print("Argmaxing...")
    for list_idx, outs in tqdm(enumerate(total_result_list)):
        total_result_list[list_idx] = np.argmax(outs.squeeze(), axis=1)

    # resize 병렬처리
    preds_array = np.empty((0, size*size), dtype=np.compat.long)
    pool = Pool(4)
    oms = pool.map(albu_transform, total_result_list)
    pool.close()
    pool.join()

    for masks in oms:
        preds_array = np.vstack((preds_array, np.squeeze(masks)))

     # PredictionString 대입
    print("Generating Submission...")
    submission = pd.DataFrame(data={'image_id': [], 'PredictionString': []})
    for file_name, string in tqdm(zip(file_names, preds_array)):
        submission = submission.append({"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
                                       ignore_index=True)

    # submission.csv로 저장
    if not os.path.exists(cfgs.output_path):
        os.makedirs(cfgs.output_path)

    submission.to_csv(
        f"{cfgs.output_path}/submission_{output_name}.csv", index=False)

    end = time.time()
    print(f"{end - start:.5f} sec")

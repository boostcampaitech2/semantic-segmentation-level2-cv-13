import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import fix_seed, arg_parse

from importlib import import_module
import tqdm
import os
import json
from collections import namedtuple
import warnings
warnings.filterwarnings("ignore")

import albumentations as A
from albumentations.pytorch import ToTensorV2

def collate_fn(batch):
    return tuple(zip(*batch))

def test(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.compat.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):
            if step % 10 == 0:
                print(f"Step: {step+1}")
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))['out']
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


if __name__ == "__main__":
    
    args = arg_parse()

    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    # seed 고정
    fix_seed(cfgs.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_name = cfgs.weight_path.split("/")[-1].replace('.pt', "")

    # model arch
    model_module = getattr(import_module("model"), cfgs.model.name)
    model = model_module(**cfgs.model.args._asdict())

    # load model weights
    checkpoint = torch.load(cfgs.weight_path, map_location=device)
    state_dict = checkpoint.state_dict()

    model.load_state_dict(state_dict)
    model = model.to(device)

    # TTA
    test_transform = A.Compose([
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensorV2()
                               ])

    test_dataset_module = getattr(import_module("dataset"), cfgs.dataset)
    test_dataset = test_dataset_module(data_root = cfgs.data_root, json_dir = cfgs.json_path,
                                       transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             **cfgs.dataloader.args._asdict(),
                             collate_fn=collate_fn)

    # sample_submisson.csv 열기
    submission = pd.read_csv(cfgs.sample_submission_path, index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    if not os.path.exists(cfgs.output_path):
        os.makedirs(cfgs.output_path)

    submission.to_csv(f"{cfgs.output_path}/submission_{output_name}.csv", index=False)
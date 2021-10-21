import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from importlib import import_module
import argparse
import tqdm
import os
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

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="FCN_resnet50")
    parser.add_argument("--weight_path", type=str, default="./results/fcn_resnet50_best_model(pretrained).pt")
    parser.add_argument("--num_classes", type=int, default = 11)
    parser.add_argument("--batch_size", type=int, default = 8)
    parser.add_argument("--output_path", type=str, default = "./submission")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    model_path = args.weight_path
    n_classes = args.num_classes
    batch_size = args.batch_size
    output_path = args.output_path
    output_name = model_path.split("/")[-1].replace('.pt', "")

    # model arch
    model_module = getattr(import_module("model"), model_name)
    model = model_module(n_classes)

    # load model weights
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()

    model.load_state_dict(state_dict)
    model = model.to(device)

    # TTA
    test_transform = A.Compose([
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensorV2()
                               ])

    test_dataset_module = getattr(import_module("dataset"), "TestDataset")
    test_dataset = test_dataset_module(data_root = "../input/data", json_dir = "../input/data/test.json",
                                       transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=4,
                             collate_fn=collate_fn)

    # sample_submisson.csv 열기
    submission = pd.read_csv(f'/opt/ml/segmentation/baseline_code/submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    submission.to_csv(f"{output_path}/submission_{output_name}.csv", index=False)
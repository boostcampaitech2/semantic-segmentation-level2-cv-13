import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from importlib import import_module

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
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):
            print(step)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './tmp/fcn_resnet50_best_model(pretrained).pt'

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()

    num_classes = 11
    
    model_module = getattr(import_module("model"), "FCN_resnet50")
    model = model_module(num_classes)

    model.load_state_dict(state_dict)
    model = model.to(device)

    test_transform = A.Compose([
                           ToTensorV2()
                           ])

    test_dataset_module = getattr(import_module("dataset"), "CustomDataSet")
    test_dataset = test_dataset_module(data_root="../input/data", json_dir="../input/data/test.json", mode = 'test', transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                                          batch_size=24,
                                          num_workers=4,
                                          collate_fn=collate_fn)

    # sample_submisson.csv 열기
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv("./submission/fcn_resnet50_best_model(pretrained).csv", index=False)
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import fix_seed, arg_parse

from importlib import import_module
from tqdm import tqdm
import os
import json
from collections import namedtuple

# tta
import ttach as tta

def collate_fn(batch):
    return tuple(zip(*batch))

def test(model, test_loader, device):
    size = 256
    tf = transforms.Compose([transforms.Resize(size)])
    print('Start prediction.')

    model.eval()
    
    file_name_list = []
    preds_array = torch.empty((0, size*size), dtype=torch.int, device=device)
    
    with torch.no_grad():
        for imgs, image_infos in tqdm(test_loader):
            # inference (512 x 512)
            imgs = torch.stack(imgs).to(device)
            outs = model(imgs)['out']
            oms = torch.argmax(outs.squeeze(), dim=1)
            
            # resize (256 x 256)
            oms = tf(oms).reshape(oms.shape[0], size*size).int()
            preds_array = torch.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    preds_array = preds_array.cpu().numpy()
    
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
    model = model_module(**cfgs.model.args._asdict()).to(device)

    # load model weights
    checkpoint = torch.load(cfgs.weight_path, map_location=device)
    state_dict = checkpoint['net']

    model.load_state_dict(state_dict)
    
    if cfgs.tta:
        tta_transform = getattr(import_module("ttach.aliases"), cfgs.tta.name)
        model = tta.SegmentationTTAWrapper(model, tta_transform(**cfgs.tta.args._asdict()), output_mask_key = 'out')

    test_augmentation_module = getattr(import_module("augmentation"), cfgs.augmentation)
    test_augmentation = test_augmentation_module().transform

    test_dataset_module = getattr(import_module("dataset"), cfgs.dataset)
    test_dataset = test_dataset_module(data_root = cfgs.data_root, json_dir = cfgs.json_path,
                                       transform=test_augmentation)
    test_loader = DataLoader(dataset=test_dataset,
                             **cfgs.dataloader.args._asdict(),
                             collate_fn=collate_fn)

    submission = pd.DataFrame(data={'image_id': [], 'PredictionString': []})
    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    print("Generating Submission...")
    for file_name, string in tqdm(zip(file_names, preds)):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    if not os.path.exists(cfgs.output_path):
        os.makedirs(cfgs.output_path)

    submission.to_csv(f"{cfgs.output_path}/submission_{output_name}.csv", index=False)
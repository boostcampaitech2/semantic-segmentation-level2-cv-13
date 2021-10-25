from collections import namedtuple
import os
import glob
import json
import shutil
import re
from importlib import import_module
from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import label_accuracy_score, add_hist, fix_seed, arg_parse
from dataset import *
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# randomness control
import numpy as np
from tqdm import tqdm

import wandb
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(rc={'figure.figsize':(12,12)})

# logging date
from datetime import datetime


category_names = ['Backgroud','General trash','Paper',
                    'Paper pack', 'Metal', 'Glass',
                    'Plastic', 'Styrofoam', 'Plastic bag',
                    'Battery', 'Clothing']
category_dicts = {k:v for k,v in enumerate(category_names)}

cur_date = datetime.today().strftime("%y%m%d")

def increment_path(path, exist_ok=False):
    """
    Automatically increment path
    
    Args:
        path (str or pathlib.Path): f"{saved_dir}/{wandb_run_name}"
        exist_ok (bool): whether to increment path (increment path if False)

    """

    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs] 
        # path.stem은 그 path에서 파일 이름에서 확장자빼고 가져옴.
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def save_checkpoint(epoch, model, loss, miou, optimizer, saved_dir, scheduler, file_name):
    check_point = {'epoch': epoch,
                    'net': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'miou': miou,
                    }
    if scheduler:
        check_point['scheduler_state_dict'] = scheduler.state_dict()

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    # load model if resume_from is set
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    start_loss = checkpoint['loss']
    prv_best_miou = checkpoint['miou']

    return model, optimizer, scheduler, start_epoch, start_loss, prv_best_miou


def collate_fn(batch):
    return tuple(zip(*batch))

def validation(epoch, num_epochs, model, data_loader, criterion, device):
    model.eval()

    example_images = []
    n_class = 11
    total_loss = 0

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    
    with torch.no_grad():
        hist = torch.zeros((n_class, n_class), device=device)
        for step, (images, masks) in pbar:
            
            images = torch.stack(images).to(device)  
            masks = torch.stack(masks).long().to(device)  
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            
            outputs = torch.argmax(outputs, dim=1)
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)

            acc, mIoU, IoU = label_accuracy_score(hist)
            avrg_loss = total_loss / (step+1)
            
            description = f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}' 
            description += f', Accuracy : {round(acc.item(), 4)}, mIoU: {round(mIoU.item(), 4)}'
            pbar.set_description(description)
            
            # 10 step마다 wandb에 이미지 로깅
            if step % 10 == 0:
                outputs = outputs.cpu().numpy()
                masks = masks.cpu().numpy()
                example_images.append(wandb.Image(
                    images[0],
                    masks = {
                        "predictions": {
                            "mask_data": outputs[0],
                            "class_labels": category_dicts
                        },
                        "ground_truth":{
                            "mask_data": masks[0],
                            "class_labels": category_dicts
                        }
                    }
                ))

    # gpu 메모리의 tensor를 cpu 메모리로 복사
    acc = acc.item()
    avrg_loss = avrg_loss.item()
    mIoU = mIoU.item()
    IoU = IoU.tolist()

    IoU_by_class = [[c,IoU] for IoU,c in zip(IoU, category_names)]
    print('IoU by class')
    for idx in range(0,len(IoU_by_class)-1,2):
        if idx != len(IoU_by_class)-3:
            print(f'{IoU_by_class[idx][0]}: {IoU_by_class[idx][1]:.4f}', end='    ')
            print(f'{IoU_by_class[idx+1][0]}: {IoU_by_class[idx+1][1]:.4f}')
        else:
            print(f'{IoU_by_class[idx][0]}: {IoU_by_class[idx][1]:.4f}', end='    ')
            print(f'{IoU_by_class[idx+1][0]}: {IoU_by_class[idx+1][1]:.4f}', end='    ')
            print(f'{IoU_by_class[idx+2][0]}: {IoU_by_class[idx+2][1]:.4f}')

    wandb.log({
        "Predicted Images with GT": example_images,
        "Validation Accuracy": round(acc,4),
        "Average Validation Loss": round(avrg_loss, 4),
        "Validation mIoU": round(mIoU, 4)
    })
    
    if epoch != num_epochs:
        return avrg_loss, mIoU
    else:
        hist = hist.cpu().numpy()
        return avrg_loss, mIoU, IoU_by_class, hist
    
    

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, 
          saved_dir, val_every, save_mode, resume_from, checkpoint_path, 
          device, scheduler = None, fp16 = False):

    print(f'Start training..')
    start_epoch = 0
    n_class = 11
    best_loss = 9999999
    best_miou = 0

    if resume_from:
        model, optimizer, scheduler, start_epoch, best_loss, best_miou = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    
    if fp16:
        print("Mixed precision is applied")
        scaler = GradScaler()

    for epoch in range(start_epoch, num_epochs):
        model.train()

        running_loss = None
        hist = torch.zeros((n_class, n_class), device=device)

        pbar = tqdm(enumerate(train_loader), total = len(train_loader))
        for step, (images, masks) in pbar:
            images = torch.stack(images).to(device)
            masks = torch.stack(masks).long().to(device) 
            
            optimizer.zero_grad()
            if fp16:
                with autocast():
                    outputs = model(images)['out']
                    loss = criterion(outputs, masks)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)['out']            
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()


            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01
            
            if isinstance(outputs, list):
                outputs = outputs[1]
            outputs = torch.argmax(outputs, dim=1)

            hist = add_hist(hist, masks, outputs, n_class=n_class)
            _, mIoU, _ = label_accuracy_score(hist)
          
            description =  f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}]: ' 
            description += f'running Loss: {round(running_loss,4)}, mIoU: {round(mIoU.item(),4)}'
            pbar.set_description(description)
            
            # 25 step마다 wandb에 loss 로깅
            if (step + 1) % 25 == 0:
                wandb.log(
                    {
                        "Train Loss": round(loss.item(), 4),
                        "Train mIoU": round(mIoU.item(),4)
                    }
                )
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            if epoch+1 < num_epochs:
                avrg_loss, miou = validation(epoch+1, num_epochs, model, val_loader, criterion, device)
            else:
                avrg_loss, miou, class_iou, hist = validation(epoch+1, num_epochs, model, val_loader, criterion, device)
            
            # save_mode에 따라 모델 저장
            if save_mode == "loss": # loss에 따라 모델 저장
                if avrg_loss < best_loss:
                    print(f"Best performance at epoch: {epoch + 1}")
                    print(f"Save model in {saved_dir}")
                    best_loss = avrg_loss
                    save_checkpoint(epoch, model, best_loss, best_miou, optimizer, saved_dir, scheduler, file_name=f"{model.model_name}_{round(best_loss,3)}_{cur_date}.pt")
                    
            else: # miou 기준 모델 저장
                if miou > best_miou:
                    print(f"Best performance at epoch: {epoch + 1}")
                    print(f"Save model in {saved_dir}")
                    best_miou = miou
                    best_loss = avrg_loss # best miou일 때도 loss tracking 수행 후 checkpoint에 저장
                    save_checkpoint(epoch, model, best_loss, best_miou, optimizer, saved_dir, scheduler, file_name=f"{model.model_name}_{round(best_miou, 3)}_{cur_date}.pt")
            
            # lr 조정
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(miou)
                else:
                    scheduler.step()

    #heatmap    
    ax = plt.subplots(figsize=(12,12))
    ax = sns.heatmap(hist/np.sum(hist, axis=1).reshape(-1,1), annot = True, cmap = 'Blues', fmt = ".4f") # gt 중에서 해당 prediction이 차지하는 비율이 얼마나 되는지
    ax.set_title("Confusion Matrix for the latest results")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")

    wandb.log(
        {
            "Confusion Matrix": wandb.Image(ax),
            "IoU by Class": wandb.plot.bar(wandb.Table(data=class_iou, columns=["label","value"]), "label","value", title="IoU by class")
        }
    )

def main():
    args = arg_parse()

    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    # fix seed
    fix_seed(cfgs.seed)
    
    # wandb logging init
    wandb.init(project=cfgs.wandb_prj_name, name=cfgs.wandb_run_name, entity="cval_seg")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # augmentation
    train_augmentation_module = getattr(import_module("augmentation"), cfgs.train_augmentation)
    train_augmentation = train_augmentation_module().transform

    val_augmentation_module = getattr(import_module("augmentation"), cfgs.val_augmentation)
    val_augmentation = val_augmentation_module().transform

    # dataset & data loader
    train_dataset_module = getattr(import_module("dataset"), cfgs.train_dataset.name)
    train_dataset = train_dataset_module(cfgs.data_root, cfgs.train_json_path, **cfgs.train_dataset.args._asdict(), transform = train_augmentation)
    train_dataloader = DataLoader(train_dataset, **cfgs.train_dataloader.args._asdict(), collate_fn=collate_fn)
    
    val_dataset_module = getattr(import_module("dataset"), cfgs.val_dataset.name)
    val_dataset = val_dataset_module(cfgs.data_root, cfgs.val_json_path, **cfgs.val_dataset.args._asdict(), transform = val_augmentation)
    val_dataloader = DataLoader(val_dataset, **cfgs.val_dataloader.args._asdict(), collate_fn = collate_fn)


    # model
    model_module = getattr(import_module("model"), cfgs.model.name)
    model = model_module(**cfgs.model.args._asdict()).to(device)

    # criterion
    if hasattr(import_module("criterions"), cfgs.criterion.name):
        criterion_module = getattr(import_module("criterions"), cfgs.criterion.name)
    else:
        criterion_module = getattr(import_module("torch.nn"), cfgs.criterion.name)
    
    criterion = criterion_module()

    # optimizer
    if hasattr(import_module("optimizers"), cfgs.optimizer.name):
        optimizer_module = getattr(import_module("optimizers"), cfgs.optimizer.name)
    else:
        optimizer_module = getattr(import_module("torch.optim"), cfgs.optimizer.name)

    optimizer = optimizer_module(model.parameters(), **cfgs.optimizer.args._asdict())


    # scheduler
    try:
        if hasattr(import_module("scheduler"), cfgs.scheduler.name):
            scheduler_module = getattr(import_module("scheduler"), cfgs.scheduler.name)
            scheduler = scheduler_module(optimizer, **cfgs.scheduler.args._asdict())
        else:
            scheduler_module = getattr(import_module("torch.optim.lr_scheduler"), cfgs.scheduler.name)
            scheduler = scheduler_module(optimizer, **cfgs.scheduler.args._asdict())
    except AttributeError :
            print('There is no Scheduler!')
            scheduler = None
    
    # get a path to save checkpoints and config
    saved_dir = increment_path(f"{cfgs.saved_dir}/{cfgs.wandb_run_name}")
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # save a config.json before training
    shutil.copy(args.cfg, f"{saved_dir}/config.json")

    # call train
    train_args = {
        'num_epochs': cfgs.num_epochs, 
        'model': model, 
        'train_loader': train_dataloader, 
        'val_loader': val_dataloader, 
        'criterion': criterion, 
        'optimizer': optimizer, 
        'saved_dir': saved_dir, 
        'val_every': cfgs.val_every, 
        'save_mode': cfgs.save_mode, 
        'resume_from': cfgs.resume_from, 
        'checkpoint_path': cfgs.checkpoint_path, # absolute path
        'device': device,
        'scheduler': scheduler,
        'fp16': cfgs.fp16
    }

    train(**train_args)
    wandb.run.finish() # close current wandb run session

if __name__ == "__main__":
    main()

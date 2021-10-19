from utils import label_accuracy_score, add_hist
from dataset import *
import torch
from torch.utils.data import DataLoader
import json
import argparse
from collections import namedtuple
from importlib import import_module

import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(rc={'figure.figsize':(12,12)})


category_names = ['Backgroud','General trash','Paper',
                    'Paper pack', 'Metal', 'Glass',
                    'Plastic', 'Styrofoam', 'Plastic bag',
                    'Battery', 'Clothing']
category_dicts = {k:v for k,v in enumerate(category_names)}

def collate_fn(batch):
    return tuple(zip(*batch))

def save_model(model, saved_dir, file_name='fcn_resnet50_best_model(pretrained).pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)

def validation(epoch, num_epochs, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    example_images = []
    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)

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
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        #IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        IoU_by_class = [[c,IoU] for IoU,c in zip(IoU, sorted_df['Categories'])]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print('IoU by class')
        for idx in range(0,len(IoU_by_class)-1,2):
            print(f'{IoU_by_class[idx][0]}: {IoU_by_class[idx][1]:.4f}', end='    ')
            print(f'{IoU_by_class[idx+1][0]}: {IoU_by_class[idx+1][1]:.4f}')

        wandb.log({
            "Predicted Images with GT": example_images,
            "Validation Accuracy": round(acc,4),
            "Average Validation Loss": round(avrg_loss.item(), 4),
            "Validation mIoU": round(mIoU, 4)
        })
        
        if epoch == num_epochs:
            return avrg_loss, IoU_by_class, hist
        
    return avrg_loss

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    
    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)['out']
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                wandb.log(
                    {
                        "Train Loss": round(loss.item(), 4),
                        "Train mIoU": round(mIoU,4)
                    }
                )
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            if epoch+1 < num_epochs:
                avrg_loss = validation(epoch+1, num_epochs, model, val_loader, criterion, device)
            else:
                avrg_loss, class_iou, hist = validation(epoch+1, num_epochs, model, val_loader, criterion, device)
            
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = avrg_loss
                save_model(model, saved_dir)

    # heatmap    
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

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    args = parser.parse_args()
    
    return args

def main():
    args = arg_parse()

    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    ## wandb logging init
    wandb.init(project=cfgs.wandb_prj_name, name=cfgs.wandb_run_name, entity="cval_seg")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
                            ToTensorV2()
                            ])
    val_transform = A.Compose([
                            ToTensorV2()
                            ])

    train_dataset_module = getattr(import_module("dataset"), cfgs.train_dataset)
    train_dataset = train_dataset_module(data_root=cfgs.data_root, json_dir=cfgs.train_json_path, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=cfgs.train_batch_size, num_workers=cfgs.num_workers, shuffle=True, collate_fn=collate_fn)

    val_dataset_module = getattr(import_module("dataset"), cfgs.val_dataset)
    val_dataset = val_dataset_module(data_root=cfgs.data_root, json_dir=cfgs.val_json_path, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=cfgs.val_batch_size, num_workers=cfgs.num_workers, shuffle=False, collate_fn=collate_fn)

    num_classes = train_dataset.num_classes

    model_module = getattr(import_module("model"), cfgs.model)
    model = model_module(num_classes).to(device)

    if hasattr(import_module("criterions"), cfgs.criterion):
        criterion_module = getattr(import_module("criterions"), cfgs.criterion)
    else:
        criterion_module = getattr(import_module("torch.nn"), cfgs.criterion)
    
    criterion = criterion_module()
    
    if hasattr(import_module("optimizers"), cfgs.criterion):
        optimizer_module = getattr(import_module("optimizers"), cfgs.optimizer)
    else:
        optimizer_module = getattr(import_module("torch.optim"), cfgs.optimizer)
    
    optimizer = optimizer_module(params = model.parameters(), lr = cfgs.lr, weight_decay=1e-6)

    train(cfgs.num_epochs, model, train_dataloader, val_dataloader, criterion, optimizer, cfgs.saved_dir, cfgs.val_every, device)

    wandb.run.finish()

if __name__ == "__main__":
    main()
import os
import monai.losses
import monai.optimizers
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import monai

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser

from tqdm.auto import tqdm
from monai.metrics import compute_dice, compute_iou, compute_hausdorff_distance
from models.MBUSNet import MBUSNet

parser = ArgumentParser()
#实例化parser，解析命令行传入的参数,default是默认参数，type是传入的类型
# parser.add_argument("--seed",type =int, default = 3045)
parser.add_argument("--seed",type =int, default = 3049)
parser.add_argument("--batch_size",type = int, default= 16)
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--epochs ",type = int, default= 150)
parser.add_argument("--img_size",type=int,default=256)
# parser.add_argument("--clip_grad",type = float,default = 5)
parser.add_argument("--weight_decay",type = float,default = 0)
# parser.add_argument("--wandb",action="store_true")

val_fold = 1
args = parser.parse_args()

class edge_loss(nn.Module): 
    def __init__(self):
        super(edge_loss, self).__init__()

        self.pool = nn.MaxPool2d(3, stride=1, padding=1)  # 用于模糊或平滑图像的边缘，以减少边缘的细节。
        self.loss = nn.MSELoss().cuda()

    def forward(self, x, target):  # x 是输入的特征图，target 是目标特征图。
        t_smoothed = self.pool(target)
        edge2 = torch.abs(target - t_smoothed).cuda()  # 计算目标特征图和模糊后目标特征图的绝对差值，并计算其绝对值。
        edge_loss = self.loss(x, edge2)

        return edge_loss

#数据类的编写
#训练参数的设置
img_size = args.img_size
lr = args.lr
batch_size = args.batch_size
n_epochs = 150
weight_decay = args.weight_decay
#损失函数为DiceLoss
criterion  = monai.losses.DiceLoss()
criterion2 = nn.BCEWithLogitsLoss()
criterion3 = edge_loss()

#训练集数据增强方式

#验证集
val_aug_list = A.Compose([
    A.Resize(img_size, img_size,interpolation=cv2.INTER_CUBIC),
],additional_targets={"ela": "image"},)

def get_transforms(type):
    if type == 'train':
        aug = tr_aug_list
    else:
        aug = val_aug_list
    return aug

#数据集的划分
dt = pd.read_csv("/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/yinhanlong/MBUS_dataset_revised/data/train_dt_5fold.csv")
#验证集
val_f1 = dt[dt.fold == val_fold].img_ids.tolist()
val_set = MMBUSDataset(val_f1, ram_cache=True,transform=get_transforms("val"))
#训练集
train_f1 = dt[dt.fold!= val_fold].img_ids.tolist()
train_set = MMBUSDataset(train_f1, ram_cache=True, transform=get_transforms("train"))

def calc_cv_(mask_gt, mask_pred):
    dice = compute_dice(mask_pred, mask_gt)
    iou = compute_iou(mask_pred, mask_gt)
    # hd = compute_hausdorff_distance(mask_pred, mask_gt, percentile=100)
    score = 0.572*dice + 0.428*iou
    return dice, iou, score

def dice_loss(logits, target):
    smooth = 1e-5
    prob  = torch.sigmoid(logits)
    batch = prob.size(0)
    prob   = prob.view(batch,1,-1)
    target = target.view(batch,1,-1)
    intersection = torch.sum(prob*target, dim=2)
    denominator  = torch.sum(prob, dim=2) + torch.sum(target, dim=2)
    dice = (2*intersection + smooth) / (denominator + smooth)
    dice = torch.mean(dice)
    dice_loss = 1. - dice
    return dice_loss

#加载DataLoader
#shuffle：在每个epoch开始的时候，对样本重新排序
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_set,batch_size=batch_size,shuffle = False)

model = MBUSNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
scheduler = StepLR(optimizer=optimizer,step_size=30,gamma=0.5)

best_loss = float("inf")
best_score = float("-inf")
best_model_path = None

#训练过程
for epoch in range(n_epochs):
        #模型训练
        model.train()
        train_loss = 0
        length = 0
        for img,ela,mask in tqdm(train_loader):
            img,ela,mask = img.to(device),ela.to(device),mask.to(device)
            output,e_1,e_2,e_3,e_4,init_out  = model(img,ela)
            loss0 = criterion(output,mask)
            loss_b1 = criterion2(e_1,mask)
            loss_b2 = criterion2(e_2,mask)
            loss_b3 = criterion2(e_3,mask)
            loss_b4 = criterion2(e_4,mask)
            loss_init = criterion2(torch.sigmoid(init_out),mask)

            loss = loss0 + loss_b1 + loss_b2 + loss_b3 + loss_b4 + 0.2*loss_init
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            length += len(mask)
            train_loss+=loss.item()
        train_loss /= len(train_loader)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

        #模型验证
        model.eval()
        valid_loss = 0
        dice_score = []
        iou_score = []
        scores = []
        mean_dice = 0
        mean_iou  = 0
        mean_score = 0
        for img,ela,mask in tqdm(val_loader):
            dice, iou, score = 0,0,0
            img,ela,mask = img.to(device),ela.to(device),mask.to(device)
            # print(f'mask:{mask}')
            score_tensor = []
            with torch.no_grad():
                output,e_1,e_2,e_3,e_4,_ = model(img,ela)
                pred_mask = (output > 0.5).float()
                dice,iou,score  = calc_cv_(mask, pred_mask)
                dice_score.append(dice)
                iou_score.append(iou)
                scores.append(score)
                loss = criterion(output, mask)
                # Record the loss and accuracy.
                valid_loss+=loss.item()

        mean_dice = torch.cat(dice_score).mean()
        mean_iou = torch.cat(iou_score).mean()
        mean_score =   torch.cat(scores).mean()  
        valid_loss /=len(val_loader)
        
        if mean_score >= best_score:
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_score = mean_score
            # torch.save(model,f"saved_models/DPV3p_model.pt")
            best_model_path = '/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/yinhanlong/MBUS_segmentation/saved_model/023_Dual_SAGATE_uaca_BD_init/fold{}_model_{}_loss{}_score{}.pth'.format(val_fold,epoch,valid_loss,best_score)
            torch.save(model.state_dict(),best_model_path)

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}")
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] dice = {mean_dice:.5f}")
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] iou = {mean_iou:.5f}")
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] score = {mean_score:.5f}")
        print("第%d轮的学习率:%f"%(epoch, optimizer.param_groups[0]['lr']))
        # wandb.log({"lr": optimizer.param_groups[0]['lr']})
        scheduler.step()

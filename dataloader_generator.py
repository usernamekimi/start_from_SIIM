from glob import glob
import pandas as pd
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import albumentations as A
import torch
import os
import shutil
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.nn import functional as F
from glob import glob
import sklearn
from torch import nn
import warnings
import gradcam

warnings.filterwarnings("ignore", category=DeprecationWarning) 

DATA_PATH = '/home/xhzhu/data/my_siim'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0")

def get_train_transforms():
    return A.Compose([
            # A.Resize(height=384, width=384, p=1),
            # A.RandomSizedCrop(min_max_height=(350, 350), height=384, width=384, p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ToGray()
            #A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),                  
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=384, width=384, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

TRAIN_ROOT_PATH = f'{DATA_PATH}/512x512-dataset-melanoma/512x512-dataset-melanoma'
df_folds = pd.read_csv(f'{DATA_PATH}/folds.csv', index_col='image_id')
set(df_folds[df_folds['fold'] == 0]['patient_id'].values).intersection(df_folds[df_folds['fold'] == 1]['patient_id'].values)

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class DatasetRetriever(Dataset):

    def __init__(self, image_ids, labels, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0

        label = self.labels[idx]

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        target = onehot(2, label)
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self):
        return list(self.labels)

from catalyst.data.sampler import BalanceClassSampler
class TrainGlobalConfig:
    num_workers = 2
    batch_size = 1 
    n_epochs = 20
    lr = 0.00003

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

#     SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
#     scheduler_params = dict(
#         max_lr=0.001,
#         epochs=n_epochs,
#         steps_per_epoch=int(len(train_dataset) / batch_size),
#         pct_start=0.1,
#         anneal_strategy='cos', 
#         final_div_factor=10**5
#     )
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.8,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )

def train_fold(fold_number):
    
    train_dataset = DatasetRetriever(
        image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
        labels=df_folds[df_folds['fold'] != fold_number].target.values,
        transforms=get_train_transforms(),
    )

    df_val = df_folds[(df_folds['fold'] == fold_number) & (df_folds['source'] == 'ISIC20')]

    validation_dataset = DatasetRetriever(
        image_ids=df_val.index.values,
        labels=df_val.target.values,
        transforms=get_valid_transforms(),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
        batch_size=TrainGlobalConfig.batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
    )
    return train_loader

    # fitter = Fitter(model=net, device=torch.device('cuda:0'), config=TrainGlobalConfig, folder=f'fold{fold_number}')
    # fitter.load(BASE_STATE_PATH)
    # fitter.fit(train_loader, val_loader)


if __name__ == '__main__':
    #for fold_number in  [1, 2, 3, 4]: # range(5)
    dataloader = train_fold(fold_number=0)
    model_path = "/home/xhzhu/mycode/siim_code/model/b3/all_foldv2/fold1/012epoch.bin"
    test_path = "/home/xhzhu/data/my_siim/512x512-dataset-melanoma/512x512-dataset-melanoma"
    save_path = "/home/xhzhu/mycode/siim_code/deep_cam_dataloader_test"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)
    gradcam.dataloader_test(model_path, dataloader, save_path, net_num=3)
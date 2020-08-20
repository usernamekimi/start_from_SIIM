'''
pytorch training baseline  
  
Using External Data: 512x512-dataset-melanoma 60000+
StratifyGroupKFold
Focal Loss / Label Smoothing (√)
BalanceClassSampler
SimpleAugs
512x512 image size

use efficientb5  LB score: 0.928 5folds (0.913,...)    

use EfficientNetb3  LB score: 0.926 5folds (0.922,0.906,...)  zxh 0.936 5fold(+tta)  2020/07/08  v1

update: single GPU, resize 512x512 into 416x416, batch_size: 8->16,  scheduler_params:patience 1->2       v2


'''

#import libraries
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
import cv2
from skimage import io
import torch
import os
from datetime import datetime
import time
import random
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.nn import functional as F
from glob import glob
import sklearn
from warmup_scheduler import GradualWarmupScheduler
from torch import nn
import warnings
from cnn_finetune import make_model


warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


#设置GPU 5和6 为可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2,5"
device_ids=range(torch.cuda.device_count())
device = torch.device("cuda:0")


DATA_PATH = '/home/xhzhu/data/my_siim'


#StratifyGroupKFold

#df_folds = pd.read_csv(f'{DATA_PATH}/folds.csv', index_col='image_id')

#use less duplicated images datasets
df_folds = pd.read_csv(f'{DATA_PATH}/folds_13062020.csv', index_col='image_id')
set(df_folds[df_folds['fold'] == 0]['patient_id'].values).intersection(df_folds[df_folds['fold'] == 1]['patient_id'].values)




#Augmentations

def get_train_transforms():
    return A.Compose([
            A.RandomSizedCrop(min_max_height=(400, 400), height=512, width=512, p=0.5),
            # A.ShiftScaleRotate(p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RandomBrightness(p=0.5),
            # A.RandomContrast(p=0.5),
            # A.RandomGamma(p=0.5),
       #     A.RGBShift(),
         #   A.CLAHE(p=1),
            # A.ToGray(),
            # A.OneOf([
            #     A.RandomContrast(),
            #     A.RandomGamma(),
            #     A.RandomBrightness(),
            # ], p=0.5),
            # A.HueSaturationValue(p=0.5),
            #A.Resize(height=512, width=512, p=1),
            # ImageNetPolicy(),
            # A.ChannelShuffle(p=0.5),
            # A.Resize(height=384, width=384, interpolation=cv2.INTER_AREA, p=1),
            #A.Cutout(num_holes=64, max_h_size=24, max_w_size=24, fill_value=0, p=0.5),
            A.Cutout(num_holes=64, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),                  
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            # A.ShiftScaleRotate(p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RandomBrightness(p=0.5),
            # A.RandomContrast(p=0.5),
            # A.RandomGamma(p=0.5),
       #     A.RGBShift(),
         #   A.CLAHE(p=1),
            # A.ToGray(),
            # A.OneOf([
            #     A.RandomContrast(),
            #     A.RandomGamma(),
            #     A.RandomBrightness(),
            # ], p=0.5),
            # A.HueSaturationValue(p=0.5),
            # A.ChannelShuffle(p=0.5),
            #A.Resize(height=384, width=384,  interpolation=cv2.INTER_AREA, p=1),
            ToTensorV2(p=1.0),
        ], p=1.0)


#Dataset
#training set path
TRAIN_ROOT_PATH = f'{DATA_PATH}/512x512-dataset-melanoma/512x512-dataset-melanoma'
# TRAIN_ROOT_PATH = f'/home/xhzhu/mycode/siim_code/all_roi_img'



def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class DatasetRetriever(Dataset):

    def __init__(self, image_ids, labels, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        print(self.image_ids.shape)
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
        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self):
        return list(self.labels)




#Metrics

from sklearn import metrics

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        # y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
        y_pred = nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,1]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        # print(self.y_true, "*********")
        # print(self.y_pred, "$$$$$$$4$$")
        self.score = sklearn.metrics.roc_auc_score(self.y_true, self.y_pred)

    @property
    def avg(self):
        return self.score

    def get_true(self):
        return self.y_true[2:]

    def get_pred(self):
        return self.y_pred[2:]

class F1Score(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0,1])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        # y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
        y_pred = nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,1]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.y_pred = np.around(self.y_pred).astype(int)
        self.score = sklearn.metrics.f1_score(self.y_true, self.y_pred)
        # print(self.y_true.shape, "YYYYYYYY")
    
    @property
    def avg(self):
        return self.score



class AccSocre(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0,1])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        # y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
        y_pred = nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,1]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.y_pred = np.around(self.y_pred).astype(int)
        self.score = sklearn.metrics.accuracy_score(self.y_true, self.y_pred)
    
    @property
    def avg(self):
        return self.score

class APScoreMeter(RocAucMeter):
    def __init__(self):
        super(APScoreMeter, self).__init__()

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        y_pred = nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,1]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = sklearn.metrics.average_precision_score(self.y_true, self.y_pred)


#Loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)




#Net

from efficientnet_pytorch import EfficientNet

def get_net():
    net = make_model('se_resnext101_32x4d', num_classes=2, pretrained=True)
    # net = EfficientNet.from_pretrained('efficientnet-b6')
    # net._fc = nn.Linear(in_features=2304, out_features=2, bias=True)
    #net._fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    #net._fc = nn.Linear(in_features=1792, out_features=2, bias=True)
    #net._fc = nn.Linear(in_features=1536, out_features=2, bias=True)
    #net._fc = nn.Linear(in_features=1408, out_features=2, bias=True)
    #net._fc = nn.Linear(in_features=1280, out_features=2, bias=True)
    #net._fc = nn.Linear(in_features=1280, out_features=2, bias=True)

    return net



net = get_net()
net = nn.DataParallel(net, device_ids=device_ids)
net = net.cuda()


#Fit

class Fitter:
    
    def __init__(self, model, device, config, folder):
        self.config = config
        self.epoch = 0

        #设置工作目录
        self.base_dir = f'./model/seresnext_512/{folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_score = 0
        self.best_loss = 10**5
        self.best_ap = 0
        
        self.model = model
        self.device = device
        self.best_true = np.array([])
        self.best_pred = np.array([])

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        # self.scheduler.step
        #self.scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=5, after_scheduler=self.scheduler)
        self.scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=6)

#         self.criterion = FocalLoss(logits=True).to(self.device)
        self.criterion = LabelSmoothing().to(self.device)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
            
            if self.epoch <= 6:
                self.scheduler_warmup.step(self.epoch)
                print(self.epoch, self.optimizer.param_groups[0]['lr'])

            t = time.time()
            summary_loss, roc_auc_scores, ap_scores , f1_scores, acc_scores = self.train_one_epoch(train_loader)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, roc_auc: {roc_auc_scores.avg:.5f},\
            acc:{acc_scores.avg:.5f}, ap: {ap_scores.avg:.5f}, f1_scores: {f1_scores.avg:.5f}, time: {(time.time() - t):.5f}')

            t = time.time()
            f_true, f_pred, summary_loss, roc_auc_scores, ap_scores , f1_scores, acc_scores= self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, roc_auc: {roc_auc_scores.avg:.5f},\
            acc:{acc_scores.avg:.5f}, ap: {ap_scores.avg:.5f}, f1_scores: {f1_scores.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_loss:
                self.best_loss = summary_loss.avg
                self.save_model(f'{self.base_dir}/best-loss-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-loss-checkpoint-*epoch.bin'))[:-2]:
                    os.remove(path)
                    
            if roc_auc_scores.avg > self.best_score:
                self.best_score = roc_auc_scores.avg
                self.save_model(f'{self.base_dir}/best-score-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-score-checkpoint-*epoch.bin'))[:-2]:
                    os.remove(path)
                self.best_true = f_true
                self.best_pred = f_pred
                    
            if ap_scores.avg > self.best_ap:
                self.best_ap = ap_scores.avg
                self.save_model(f'{self.base_dir}/best-ap-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-ap-checkpoint-*epoch.bin'))[:-2]:
                    os.remove(path)

            if self.config.validation_scheduler:
                if self.epoch > 6:
                    self.scheduler.step(metrics=summary_loss.avg)
          
            self.epoch += 1
        #if self.epoch == self.config.n_epochs:
        return self.best_true , self.best_pred

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        roc_auc_scores = RocAucMeter()
        ap_scores = APScoreMeter()
        f1_scores = F1Score()
        acc_scores = AccSocre()
        t = time.time()
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, roc_auc: {roc_auc_scores.avg:.5f}, ap: {ap_scores.avg:.5f} ' + \
                        f'f1_scores: {f1_scores.avg:.5f} ' + \
                        f'acc_scores: {acc_scores.avg:.5f} ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                roc_auc_scores.update(targets, outputs)
                ap_scores.update(targets, outputs)
                f1_scores.update(targets, outputs)
                acc_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)
        f_true = roc_auc_scores.get_true()
        f_pred = roc_auc_scores.get_pred()

        return f_true, f_pred, summary_loss, roc_auc_scores, ap_scores, f1_scores, acc_scores

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        roc_auc_scores = RocAucMeter()
        ap_scores = APScoreMeter()
        f1_scores = F1Score()
        acc_scores = AccSocre()
        t = time.time()
        
        # print(len(train_loader), "gggggggg")
        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, roc_auc: {roc_auc_scores.avg:.5f}, ap: {ap_scores.avg:.5f} ' + \
                        f'f1_scores: {f1_scores.avg:.5f} ' + \
                        f'acc_scores: {acc_scores.avg:.5f} ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            roc_auc_scores.update(targets, outputs)
            ap_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)
            f1_scores.update(targets, outputs)
            acc_scores.update(targets, outputs)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss, roc_auc_scores, ap_scores, f1_scores, acc_scores
    
    def save_model(self, path):
        self.model.eval()
        torch.save(self.model.state_dict(),path)

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': self.best_score,
            'best_ap': self.best_ap,
            'best_loss': self.best_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_score = checkpoint['best_score']
        self.best_ap = checkpoint['best_ap']
        self.best_loss = checkpoint['best_loss']
        self.epoch = checkpoint['epoch']
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')




#training config

class TrainGlobalConfig:
    num_workers = 2
    #batch_size = 8   
    #修改batch_size=16
    batch_size = 16
    n_epochs = 30
    lr =  0.00003

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
        patience=2,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------




#Save all states for "honest" training of folds

fitter = Fitter(model=net, device=torch.device('cuda:0'), config=TrainGlobalConfig, folder='base_state')
BASE_STATE_PATH = f'{fitter.base_dir}/base_state.bin'
fitter.save(BASE_STATE_PATH)



from catalyst.data.sampler import BalanceClassSampler

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

    fitter = Fitter(model=net, device=torch.device('cuda:0'), config=TrainGlobalConfig, folder=f'fold{fold_number}')
    fitter.load(BASE_STATE_PATH)
    f_true, f_pred = fitter.fit(train_loader, val_loader)
    return f_true, f_pred




import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

all_true = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
all_pred = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]

for fold_number in range(5):      # range(5)
    all_true[fold_number], all_pred[fold_number] = train_fold(fold_number=fold_number)
    print(all_true[fold_number].shape, all_pred[fold_number].shape)

final_true = np.hstack((all_true[0], all_true[1], all_true[2], all_true[3], all_true[4]))
final_pred = np.hstack((all_pred[0], all_pred[1], all_pred[2], all_pred[3], all_pred[4]))
print(final_true.shape, final_pred.shape)


K = pd.DataFrame(columns = ["true", "pred"])
K['true'] = final_true
K['pred'] = final_pred
K.to_csv('/home/xhzhu/mycode/siim_code/model/seresnext_512/oof.csv',index=False)
al_auc = sklearn.metrics.roc_auc_score(final_true, final_pred)
print("all_roc_auc is:", al_auc)
    





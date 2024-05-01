import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import ssl


class Dataset(BaseDataset):
   
   #所有類別，本實驗只有氣管一類
    CLASSES = ['trachea', ]

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        #STR TO CLASS 
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        #強化&前處理
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_dir = self.masks_fps[i].replace("jpg", "png")
        mask_dir = mask_dir.replace("img", "mask")
        
        mask = cv2.imread(mask_dir, 0)

        
        # 取出特定的類別 (e.g. 氣管)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # 圖像增強
        if self.augmentation:
            #print(str(np.shape(image)) + "前")
            sample = self.augmentation(image=image, mask=mask)
            #print(str(np.shape(sample['image'])) + "後")
            image, mask = sample['image'], sample['mask']

        # 圖像前處理
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

# ---------------------------------------------------------------

def get_training_augmentation():
    train_transform = [
        #由於X光圖通常有高對比及較少的躁點，且診斷時需要保留原本的樣貌
        #只使用銳化及平移10%
        albu.OneOf(
            [
                albu.Sharpen(p=1),
            ],
            p=0.9,
        ),
        albu.ShiftScaleRotate(shift_limit=0.1, rotate_limit=0, scale_limit=0, p=1),
        albu.Resize(height=256, width=256, always_apply=True),
    ]
    return albu.Compose(train_transform)



def get_validation_augmentation():
    #調整至256x256
    test_transform = [
        albu.Resize(256,256, always_apply=True)
    ]
    return albu.Compose(test_transform)



#2 0 1表示將原本的height、width、channels變為chw
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    #前處理
    #preprocessing_fn (callbale): 規範後的函數

    _transform = [
        #將圖缩放到 [0, 1] 内
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]

    return albu.Compose(_transform)

# 創建model並訓練
# ---------------------------------------------------------------
if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context

    DATA_DIR = r'./ETT_v3/Fold1/'

    # 訓練
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    # 驗證
    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')
    
    ENCODER = 'se_resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['trachea']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda' #device

    # 用已有MODEL建立分割模型

    # unet++
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
     # unet
#     model = smp.Unet(
#         encoder_name=ENCODER,
#         encoder_weights=ENCODER_WEIGHTS,
#         classes=len(CLASSES),
#         activation=ACTIVATION,
#     )
     # DeepLabV3
#     model = smp.DeepLabV3(
#         encoder_name=ENCODER,
#         encoder_weights=ENCODER_WEIGHTS,
#         classes=len(CLASSES),
#         activation=ACTIVATION,
#     )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # 載入訓練集
    train_dataset = Dataset(
        x_train_dir, #訓練集路徑
        y_train_dir, #訓練集MASK路徑
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # 載入驗證集
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # batch_size：每個批次中的樣本數量
    # shuffle：是否對數據進行隨機重排
    # num_workers：用於加載數據的線程數量
    # drop_last：如果數據樣本數不能被 batch_size 整除，TRUE丟

    #Unet 設定  Unet++ 設定 
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    
#     train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
#     valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    
    
    loss = smp.utils.losses.DiceLoss()
#     loss = smp.utils.losses.JaccardLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5)
    ]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    # 用來迭代數據樣本
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # 模型訓練參數
    max_score = 0
    patience = 10
    counter = 0

    for i in range(0, 200):

        print('\nEpoch: {}'.format(i))

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # 每次迭代保存訓練最好的模型
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping after {} epochs'.format(i))
                break

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
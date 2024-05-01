import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset as BaseDataset



# ---------------------------------------------------------------

class Dataset(BaseDataset):

    # 所有類別，本實驗只有氣管一類
    CLASSES = ['trachea']

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

        # STR TO CLASS
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        # 強化&前處理
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

        origin_image_x = image.shape[1]
        origin_image_y = image.shape[0]
        
        # 圖像增強
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # 圖像前處理
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask, origin_image_x, origin_image_y

    def __len__(self):
        return len(self.ids)

# ---------------------------------------------------------------    
# 平均誤差公分

# 換成公分
# y = 真實值
# g = 預測值
def average_error_in_centimeters(mask):
    pixel_size_cm = 1 / 72  # 每72像素一公分

    # 0的位置為索引
    zero_indices = np.argwhere(mask == 0)

    if zero_indices.size == 0:
        return None  # 沒有0則返回None

    # 找到最後一個0的位置
    last_zero_index = zero_indices[-1]
    
    print(last_zero_index)
    # 計算0的位置的實際尺寸
    answer = [last_zero_index[0] * pixel_size_cm, last_zero_index[1] * pixel_size_cm, mask[last_zero_index[0], last_zero_index[1]]]
    return answer
    
# 計算每張圖片的公分誤差
def calculate_centimeter_error(gt_mask,pr_mask):
    # 取出 Y
    gt_y = gt_mask[1]

    try:
        pr_y = pr_mask[1]
    except TypeError:
        pr_y = 0
    
    print("pr_y: ",pr_y)

    # 計算距離
    distance = abs(gt_y-pr_y)
    print("distance:", distance)
    return distance
    
# 沒有換公分，是pixel    
def average_error_in_pixel(mask):
    # 建立一個與mask相同的
    mask_pixel = np.empty((mask.shape[0], mask.shape[1], 3))
    # 轉為實際尺寸
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 0:
                answer = [i, j, mask[i,j]]
            mask_pixel[i, j] = [i, j, mask[i,j]]

    return mask_pixel, answer

# ---------------------------------------------------------------

def get_validation_augmentation():
    # 調整至256x256
    test_transform = [
        albu.Resize(256,256)
    ]
    return albu.Compose(test_transform)

# 2 0 1表示將原本的height、width、channels變為chw
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    
    # 前處理
    # preprocessing_fn (callbale): 規範後的函數

    _transform = [
        # 將圖缩放到 [0, 1] 内
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# 顯示圖像結果
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# ---------------------------------------------------------------
if __name__ == '__main__':

    DATA_DIR = r'./ETT_v3/Fold5/'

    # 測試
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    ENCODER = 'se_resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['trachea']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda' # device

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # ---------------------------------------------------------------

    # 載入訓練最佳模型
    best_model = torch.load('./best.pth')

    # 載入測試集
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # ---------------------------------------------------------------
    # 顯示分割結果
    # 顯示沒有轉化的測試集
    test_dataset_vis = Dataset(
        x_test_dir, y_test_dir,
        classes=CLASSES,
    )
    # 隨機測試
    k = 47
    total_centimeter = 0
    half = 0
    one = 0
    for i in range(k):
#         n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[i][0].astype('uint8')
        image, gt_mask, x,y  = test_dataset[i]
        gt_mask = gt_mask.squeeze()
        
        # 公式2，3
#         gt_first_white_pixel, gt_pixel_answer = average_error_in_pixel(gt_mask)
  
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
        gt_mask = cv2.resize(gt_mask, (x, y))
        pr_mask = cv2.resize(pr_mask, (x, y))
        
        # 公式1 gt 換成公分
        gt_answer = average_error_in_centimeters(gt_mask)
        
        # 公式1 pr 換成公分
        pr_answer = average_error_in_centimeters(pr_mask)
        
        # 公式2，3
#         pr_first_white_pixel, pr_pixel_answer = average_error_in_pixel(pr_mask)
        
        # 總誤差公分 Yi−Gi
        total_centimeter += calculate_centimeter_error(gt_answer, pr_answer)
        
        centimeter = calculate_centimeter_error(gt_answer, pr_answer)
        print("誤差: ",centimeter)
        
        # 誤差0.5
        if centimeter<=0.5:
            half+=1
        # 誤差1.0
        if centimeter<=1:
            one+=1

        print(half)
        print(one)

        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )
    
    # 平均誤差公分  1/K
    k_p = 1/k
    Average_error_in_centimeters = k_p*total_centimeter
    print("平均誤差公分" + str(Average_error_in_centimeters))
    print("誤差在0.5cm內準確率" + str((1/k*half)*100) + "%")
    print("誤差在1cm內準確率" + str((1/k*one)*100) + "%")
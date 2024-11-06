import os
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from torchvision import transforms
import torch.nn.functional as F

class MattingDataset(Dataset):
    def __init__(self, image_dir, trimap_dir, alpha_dir, mode):
        """
        初始化数据集
        :param image_dir: 图像文件夹路径
        :param trimap_dir: trimap文件夹路径
        :param alpha_dir: alpha文件夹路径
        :param transform: 数据预处理
        """
        self.mode=mode
        if self.mode=='train':
            self.image_dir = image_dir
            self.trimap_dir = trimap_dir
            self.alpha_dir = alpha_dir
        else:
            self.image_dir = image_dir.replace('train','test')
            self.trimap_dir = trimap_dir.replace('train','test')
            self.alpha_dir = alpha_dir.replace('train','test')
        
        self.image_names = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_names)

    def preprocess(self, image, alpha, trimap):
        transform = A.Compose([
                A.Resize(512, 512),
                ToTensorV2()
            ])
        transformed = transform(image=image, masks=[alpha, trimap])

        # 提取结果并分别赋值
        image = transformed['image']
        alpha, trimap = transformed['masks'] 

        alpha = alpha.unsqueeze(0) 
        trimap = trimap.unsqueeze(0)

        return image, alpha, trimap

    def __getitem__(self, idx):
        """
        获取图像、trimap和alpha matte的样本
        :param idx: 索引
        :return: 预处理后的图像，trimap和alpha matte
        """
        img_name = self.image_names[idx]
        base_name, _ = os.path.splitext(img_name)

        image_path = os.path.join(self.image_dir, img_name)
        trimap_path = os.path.join(self.trimap_dir, base_name + '.png')
        
        alpha_path = os.path.join(self.alpha_dir, base_name + '.png')
        

        image = cv2.imread(image_path)
        trimap = cv2.imread(trimap_path, 0)
        alpha = cv2.imread(alpha_path, 0)

        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 255
        trimap[(trimap >= 85) & (trimap < 170)] = 128
        
        image = image / 255.
        alpha = alpha / 255.
        trimap = trimap / 255.



        image, alpha, trimap = self.preprocess(image, alpha, trimap)
        image=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        return image.float(), alpha.float(), trimap.float()

if __name__ == "__main__":
    torch.set_printoptions(profile="full",precision=3)
    dataset = MattingDataset(
        image_dir="data/dis646/train/img",   # 替换为你的图像文件夹路径
        trimap_dir="data/dis646/train/trimap",     # 替换为你的 trimap 文件夹路径
        alpha_dir="data/dis646/train/alpha",   # 替换为你的 alpha 文件夹路径
        mode='train'
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    print(len(dataset))
    # 迭代读取数据
    for batch_idx, (images, alphas, trimap) in enumerate(dataloader):
        print(trimap)
        # print(images.shape, trimap.shape, alphas.shape)
        break
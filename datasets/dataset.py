#-*-coding:utf-8-*-
from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
import numpy as np
import PIL.Image as Image
from tqdm import tqdm

class Dataset_OCR(Dataset):
    def __init__(self, images_root_dir,images_name, transform=None):
        self.images_root_dir = images_root_dir
        self.transform = transform
        self.images_name=images_name

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images_name)

    def __getitem__(self, index):
        image_index = self.images_name[index]
        img_path = os.path.join(self.images_root_dir, image_index)
        img=Image.open(img_path)
        if self.transform:
            img = self.transform(img)  # 对样本进行变换
        return img,index # 返回该样本

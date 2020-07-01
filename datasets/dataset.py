#-*-coding:utf-8-*-
from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
import cv2
class Dataset_OCR(Dataset):
    '''
    define the dataset for rcnn models
    '''
    def __init__(self, images_root_dir,images_name, transform=None):
        '''
        define the dataset
        @param images_root_dir: the images root
        @param images_name: the images name of the dataset
        @param transform: the transform for the images
        '''
        self.images_root_dir = images_root_dir
        self.transform = transform
        self.images_name=images_name

    def __len__(self):
        '''
        the len of the dataset
        @return:
        '''
        return len(self.images_name)

    def __getitem__(self, index):
        image_index = self.images_name[index]
        img_path = os.path.join(self.images_root_dir, image_index)

        img=Image.open(img_path)
        img=img.convert('L')
        #print(img)
        if self.transform:
            img = self.transform(img)  # transform the image
        return img,index

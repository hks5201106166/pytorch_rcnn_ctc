#-*-coding:utf-8-*-
from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
import cv2
import time
class Dataset_OCR(Dataset):
    '''
    define the dataset for rcnn models
    '''
    def __init__(self, images_root_dir,images_name,config,transform=None):
        '''
        define the dataset
        @param images_root_dir: the images root
        @param images_name: the images name of the dataset
        @param transform: the transform for the images
        '''
        self.images_root_dir = images_root_dir
        self.transform = transform
        self.images_name=images_name
        self.idcard_text=['xingming_rect','dizhi_rect','xingbie_rect',
                         'mingzhu_rect','shengfengzhenghao_rect',
                          'chusheng_rect_year','chusheng_rect_month',
                          'chusheng_rect_day','qianfajiguang_rect',
                          'youxiaoqixian_rect']
        self.config=config
        self.count_the_same_text=len(os.listdir(os.path.join(self.images_root_dir, self.images_name[0])))/10

    def __len__(self):
        '''
        the len of the dataset
        @return:
        '''
        return len(self.images_name)
    def image_resize(self,img):

        h, w = img.shape
        h_cur = h / 32
        w_cvt1 = w / h_cur
        w_cur = 280 / 160
        w_cvt2 = int(w_cvt1 / w_cur)
        img = cv2.resize(img, dsize=(w_cvt2, 32), interpolation=2)
        img = np.reshape(img, (self.config.DATASET.IMAGE_SIZE.H, w_cvt2, 1))

        # # normalize
        img = img.astype(np.float32)
        img = (img / 255. -0.5) / 0.5
        img = img.transpose([2, 0, 1])
        img = torch.from_numpy(img)
        #img = img.view(1, *img.size())
        return img
    def __getitem__(self, index):

        image_index = self.images_name[index]
        imgs_path = os.listdir(os.path.join(self.images_root_dir, image_index))

        xingming_rect=cv2.imread(self.images_root_dir+'/'+image_index+'/'+'xingming_rect'+'-'+str(np.random.randint(0,self.count_the_same_text))+'.jpg',0)
        xingming_rect=self.image_resize(xingming_rect)
        #cv2.imshow('xingming_rect',xingming_rect)

        xingbie_rect = cv2.imread(self.images_root_dir + '/' + image_index + '/' + 'xingbie_rect' + '-' + str(
            np.random.randint(0, self.count_the_same_text)) + '.jpg', 0)
        xingbie_rect = self.image_resize(xingbie_rect)
        #cv2.imshow('xingbie_rect',xingbie_rect)

        mingzhu_rect = cv2.imread(self.images_root_dir + '/' + image_index + '/' + 'mingzhu_rect' + '-' + str(
            np.random.randint(0, self.count_the_same_text)) + '.jpg', 0)
        mingzhu_rect = self.image_resize(mingzhu_rect)
       # cv2.imshow('mingzhu_rect', mingzhu_rect)

        shengfengzhenghao_rect = cv2.imread(self.images_root_dir + '/' + image_index + '/' + 'shengfengzhenghao_rect' + '-' + str(
            np.random.randint(0, self.count_the_same_text)) + '.jpg', 0)
        shengfengzhenghao_rect = self.image_resize(shengfengzhenghao_rect)
        #cv2.imshow('shengfengzhenghao_rect', shengfengzhenghao_rect)

        chusheng_rect_year = cv2.imread(self.images_root_dir + '/' + image_index + '/' + 'chusheng_rect_year' + '-' + str(
            np.random.randint(0, self.count_the_same_text)) + '.jpg', 0)
        chusheng_rect_year = self.image_resize(chusheng_rect_year)
       # cv2.imshow('chusheng_rect_year',chusheng_rect_year)

        chusheng_rect_month = cv2.imread(
            self.images_root_dir + '/' + image_index + '/' + 'chusheng_rect_month' + '-' + str(
                np.random.randint(0, self.count_the_same_text)) + '.jpg', 0)
        chusheng_rect_month = self.image_resize(chusheng_rect_month)
       # cv2.imshow('chusheng_rect_month', chusheng_rect_month)

        chusheng_rect_day = cv2.imread(
            self.images_root_dir + '/' + image_index + '/' + 'chusheng_rect_day' + '-' + str(
                np.random.randint(0, self.count_the_same_text)) + '.jpg', 0)
        chusheng_rect_day = self.image_resize(chusheng_rect_day)
       # cv2.imshow('chusheng_rect_day', chusheng_rect_day)

        qianfajiguang_rect = cv2.imread(
            self.images_root_dir + '/' + image_index + '/' + 'qianfajiguang_rect' + '-' + str(
                np.random.randint(0, self.count_the_same_text)) + '.jpg', 0)
        qianfajiguang_rect = self.image_resize(qianfajiguang_rect)
       # cv2.imshow('qianfajiguang_rect', qianfajiguang_rect)

        youxiaoqixian_rect = cv2.imread(
            self.images_root_dir + '/' + image_index + '/' + 'youxiaoqixian_rect' + '-' + str(
                np.random.randint(0, self.count_the_same_text)) + '.jpg', 0)
        youxiaoqixian_rect = self.image_resize(youxiaoqixian_rect)
      #  cv2.imshow('youxiaoqixian_rect',youxiaoqixian_rect)

        dizhi_rect = cv2.imread(
            self.images_root_dir + '/' + image_index + '/' + 'dizhi_rect' + '-' + str(
                np.random.randint(0, self.count_the_same_text)) + '.jpg', 0)
        dizhi_rect = self.image_resize(dizhi_rect)
       # cv2.imshow('dizhi_rect', dizhi_rect)
       # cv2.waitKey(0)


        # if self.transform:
        #     img = self.transform(img)  # transform the image
        return xingming_rect,dizhi_rect,xingbie_rect,mingzhu_rect,shengfengzhenghao_rect,chusheng_rect_year,chusheng_rect_month,chusheng_rect_day,qianfajiguang_rect,youxiaoqixian_rect,image_index

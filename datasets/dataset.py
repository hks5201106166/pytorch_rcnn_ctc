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
import imgaug as ia
import imgaug.augmenters as iaa
class Dataset_OCR(Dataset):
    '''
    define the dataset for rcnn models
    '''
    def __init__(self, images_root_dir,labels,config,transform=None):
        '''
        define the dataset
        @param images_root_dir: the images root
        @param images_name: the images name of the dataset
        @param transform: the transform for the images
        '''
        self.images_root_dir = images_root_dir
        self.transform = transform
        self.config = config
        self.labels_train = labels
        self.seq = iaa.Sequential([
            iaa.OneOf([
                iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 3.0
            ]),
            iaa.MultiplyBrightness((0.5, 1.)), ], random_order=True)

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


       # cv2.imshow('dizhi_rect', dizhi_rect)
       # cv2.waitKey(0)


        # if self.transform:
        #     img = self.transform(img)  # transform the image
        return None
class Dataset_OCR_Train(Dataset):
    '''
    define the dataset for rcnn models
    '''
    def __init__(self, images_root_dir,config,labels_train,transform):
        '''
        define the dataset
        @param images_root_dir: the images root
        @param images_name: the images name of the dataset
        @param transform: the transform for the images
        '''
        self.images_root_dir = images_root_dir
        self.transform = transform
        self.config=config
        self.labels_train=labels_train
        self.seq = iaa.Sequential([
        iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                ]),
        iaa.MultiplyBrightness((0.5, 1.)),],random_order=True)



    def __len__(self):
        '''
        the len of the dataset
        @return:
        '''
        return len(self.labels_train['1'])
    def image_resize(self,img,i):
        if i==1:
            img = self.seq(image=img)
            img = cv2.cvtColor(img,code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(32, 32), interpolation=2)
            img = np.reshape(img, (32,32, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==2:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(65,32), interpolation=2)
            img = np.reshape(img, (32, 65, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==3:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(90,32), interpolation=2)
            img = np.reshape(img, (32, 90, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==4:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(120,32), interpolation=2)
            img = np.reshape(img, (32,120, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==5:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(150,32), interpolation=2)
            img = np.reshape(img, (32,150, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==6:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(170,32), interpolation=2)
            img = np.reshape(img, (32,170, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==7:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(200,32), interpolation=2)
            img = np.reshape(img, (32,200, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==8:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(217,32), interpolation=2)
            img = np.reshape(img, (32,217, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==9:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(238,32), interpolation=2)
            img = np.reshape(img, (32,238, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==10:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(270,32), interpolation=2)
            img = np.reshape(img, (32,270, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==11:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(280,32), interpolation=2)
            img = np.reshape(img, (32,280, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==12:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(305,32), interpolation=2)

            img = np.reshape(img, (32,305, 1))


            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==13:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(328,32), interpolation=2)
            img = np.reshape(img, (32,328, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==14:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(345,32), interpolation=2)
            img = np.reshape(img, (32,345, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==15:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(360,32), interpolation=2)
            img = np.reshape(img, (32,360, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==16:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(375,32), interpolation=2)
            img = np.reshape(img, (32,375, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==17:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(400,32), interpolation=2)
            img = np.reshape(img, (32,400, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==18:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, dsize=(415,32), interpolation=2)
            img = np.reshape(img, (32,415, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)
        elif i==19:
            img = self.seq(image=img)
            img = cv2.cvtColor(img, code=cv2.COLOR_RGB2GRAY)
            # cv2.imshow("hhks",img)
            # cv2.waitKey(1000)
            img = cv2.resize(img, dsize=(420,32), interpolation=2)
            img = np.reshape(img, (32,420, 1))

            # # normalize
            img = img.astype(np.float32)
            img = (img / 255. -0.5) / 0.5
            img = img.transpose([2, 0, 1])
            img = torch.from_numpy(img)

        return img
    def __getitem__(self, index):
        labels_list=[]
        images=[]
        for i in range(1,20):
            labels=self.labels_train[str(i)]
            label=labels[np.random.randint(0,len(labels))].strip()
            image_name=label.split(' ')[0]
            image=cv2.imread(os.path.join(self.images_root_dir,image_name))
            image=self.image_resize(image,i)
            images.append(image)
            labels_list.append(label)

        return images[0],images[1],images[2],images[3],images[4],images[5],images[6],\
               images[7],images[8],images[9],images[10],images[11],images[12],\
               images[13],images[14],images[15],images[16],images[17],images[18],\
               labels_list[0],labels_list[1],labels_list[2],labels_list[3],labels_list[4] ,\
               labels_list[5],labels_list[6],labels_list[7],labels_list[8],labels_list[9], \
               labels_list[10], labels_list[11], labels_list[12], labels_list[13], labels_list[14], \
               labels_list[15], labels_list[16], labels_list[17], labels_list[18]



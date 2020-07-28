#-*-coding:utf-8-*-
from datasets.dataset import Dataset_OCR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from models import crnn
import torch.nn as nn
import numpy as np
from utils.tool import LabelTool,get_logger,get_output_dir
from easydict import EasyDict as edict
import yaml
import argparse
import os
from utils.tool import train_one_epoch,Avgloss,validate
def config_args():
    '''
    define the config
    '''
    parser = argparse.ArgumentParser()
    # the config for the train
    parser.add_argument('--config',default='config/config.yaml',type=str,help='the path of the images dir')
    args = parser.parse_args()
    with open(args.config,'r') as file:
        config=yaml.load(file)
        config=edict(config)
    return config
def main():
    config = config_args()
    save_outputs=get_output_dir(config.MODEL.BACKBONE+'_'+'lstm-layer:'+str(config.MODEL.LSTM_NUM_LAYER)+'_lstm-hidden-nums:'+str(config.MODEL.LSTM_NUM_HIDDEN))
    logger=get_logger(save_outputs)
    #define the train and val transform
    transform_train=\
        transforms.Compose([
            transforms.Resize(size=(config.DATASET.IMAGE_SIZE.H,config.DATASET.IMAGE_SIZE.W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],std=[0.5])])
    transform_val=\
        transforms.Compose([
            transforms.Resize(size=(config.DATASET.IMAGE_SIZE.H,config.DATASET.IMAGE_SIZE.W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],std=[0.5])])



    #define the rcnn model
    model=crnn.CRNN(config).to(torch.device("cuda:"+config.CUDNN.GPU))
    if config.TRAIN.RESUME.IS_RESUME==True:
        print('model load .......................')
        # model_dict = model.state_dict()
        #
        # pretrained_dict = torch.load(config.TRAIN.RESUME.MODEL_SAVE)['state_dict']
        #
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['embeding.weight','embeding.bias']}
        #
        # model_dict.update(pretrained_dict)
        #
        # model.load_state_dict(model_dict)

        model.load_state_dict(torch.load(config.TRAIN.RESUME.MODEL_SAVE)['state_dict'],strict=False)

    criterion = nn.CTCLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=config.TRAIN.LR)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_STEP,gamma=config.TRAIN.LR_FACTOR)

    #get the train and val lables from the train.txt and test.txt
    label_tool = LabelTool(char_std_path=config.DATASET.CHAR_FILE)
    images_train,labels_train=label_tool.get_labels(labels_path=config.DATASET.LABELS_FILE.TRAIN)
    images_val,labels_val=label_tool.get_labels(labels_path=config.DATASET.LABELS_FILE.VAL)

    #define the train and val dataset
    dataset_train=Dataset_OCR(images_root_dir=config.DATASET.IMAGE_ROOT,images_name=images_train,transform=transform_train)
    dataset_val=Dataset_OCR(images_root_dir=config.DATASET.IMAGE_ROOT,images_name=images_val,transform=transform_val)

    #define the train and val dataloader
    dataloader_train=DataLoader(dataset=dataset_train,batch_size=config.TRAIN.BATCH,shuffle=config.TRAIN.SHUFFLE,num_workers=config.TRAIN.WORKERS)
    dataloader_val=DataLoader(dataset=dataset_val,batch_size=config.TEST.BATCH,shuffle=config.TEST.SHUFFLE,num_workers=config.TEST.WORKERS)

    avgloss=Avgloss()  #define the avgloss class
    for epoch in range(config.TRAIN.EPOCH):
        #train the rcnn models
        train_one_epoch(epoch,dataloader_train,config,model,label_tool, labels_train,criterion,avgloss,optimizer,scheduler,logger)
        #update the learn lr
        scheduler.step()

        #validate the rcnn models
        print('validate the model,please hold on:')
        validate(epoch,dataloader_val,labels_val,config,model,label_tool,criterion,save_outputs,logger)

if __name__=='__main__':
    main()
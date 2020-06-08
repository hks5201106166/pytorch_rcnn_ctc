#-*-coding:utf-8-*-
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
from models.cnn.densenet import densenet121
from torchvision.models.resnet  import resnet18
def config_args():
    parser = argparse.ArgumentParser()
    # the config for the train
    parser.add_argument('--config',default='datasets/config/config.yaml',type=str,help='the path of the images dir')
    args = parser.parse_args()
    with open(args.config,'r') as file:
        config=yaml.load(file)
        config=edict(config)
    return config
def main():
    save_outputs=get_output_dir('dev_lstm_1layer_128')
    logger=get_logger(save_outputs)
    config=config_args()
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

    criterion=nn.CTCLoss()
    label_tool=LabelTool(char_std_path=config.DATASET.CHAR_FILE)

    model=densenet121().to(torch.device("cuda:"+config.CUDNN.GPU))
    if config.TRAIN.RESUME.IS_RESUME==True:
        model.load_state_dict(torch.load(config.TRAIN.RESUME.MODEL_SAVE)['state_dict'])


    optimizer=torch.optim.Adam(model.parameters(),lr=config.TRAIN.LR)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_STEP,gamma=config.TRAIN.LR_FACTOR)

    images_train,labels_train=label_tool.get_labels(labels_path=config.DATASET.LABELS_FILE.TRAIN)
    images_val,labels_val=label_tool.get_labels(labels_path=config.DATASET.LABELS_FILE.VAL)
    dataset_train=Dataset_OCR(images_root_dir=config.DATASET.IMAGE_ROOT,images_name=images_train,transform=transform_train)
    dataset_val=Dataset_OCR(images_root_dir=config.DATASET.IMAGE_ROOT,images_name=images_val,transform=transform_val)

    dataloader_train=DataLoader(dataset=dataset_train,batch_size=config.TRAIN.BATCH,shuffle=config.TRAIN.SHUFFLE,num_workers=config.TRAIN.WORKERS)
    dataloader_val=DataLoader(dataset=dataset_val,batch_size=config.TEST.BATCH,shuffle=config.TEST.SHUFFLE,num_workers=config.TEST.WORKERS)

    step=0
    loss_all=0
    for epoch in range(config.TRAIN.EPOCH):
        for images,indexs in dataloader_train:
            images=images.to(torch.device("cuda:"+config.CUDNN.GPU))

            output=model(images)
            sequence_len=output.shape[0]
            target, input_lengths, target_lengths = label_tool.convert_ctcloss_labels(indexs, labels_train,sequence_len)
            loss=criterion(output.cpu(),target,input_lengths,target_lengths)
            #loss_all+=loss.cpu().detach().numpy()
            loss_all += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(scheduler.get_lr())
            step+=1
            if step%config.TRAIN.SHOW_STEP==0:
                print("epoch:{},step:{},loss={},loss_avarage={},lr={}".format(epoch,step,loss,loss_all/step,scheduler.get_lr()[0]))
                logger.debug("epoch:{},step:{},loss={},loss_avarage={},lr={}".format(epoch,step,loss,loss_all/step,scheduler.get_lr()[0]))
        scheduler.step()
        loss_all_val=0
        step_val=0
        nums_all=0
        nums_all_correct=0
        for images_val,indexs_val in dataloader_val:
            images_val = images_val.to(torch.device("cuda:"+config.CUDNN.GPU))
            with torch.no_grad():
                output_val = model(images_val)
            sequence_len_val = output_val.shape[0]
            target_val, input_lengths_val, target_lengths_val = label_tool.convert_ctcloss_labels(indexs_val,
                                                                                                  labels_val,
                                                                                                  sequence_len_val)
            preds_val = output_val.permute(1, 0, 2).argmax(2).cpu().numpy()
            preds_str_val,preds_str_val_blank = label_tool.decode_batch(preds_val)
            correct_nums = label_tool.cal_correct_nums(preds_str_val,preds_str_val_blank,indexs_val, labels_val,step_val)
            nums_all_correct+=correct_nums
            nums_all+=output_val.shape[1]
            print('nums_all_correct{},nums_all{}'.format(nums_all_correct,nums_all))

            loss_val = criterion(output_val, target_val, input_lengths_val, target_lengths_val)
            loss_all_val += loss_val
            step_val+=1
        acc=nums_all_correct/nums_all
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "acc": acc,
            }, os.path.join('save_outputs/'+save_outputs+'/models_saved/', "epoch_{}_acc_{:.4f}.pth".format(epoch, acc))
        )
        print("epoch:{},val_loss_avarage={},val_accuracy={}".format(epoch,loss_all_val / step_val,nums_all_correct/nums_all))
        logger.debug("epoch:{},val_loss_avarage={},val_accuracy={}".format(epoch,loss_all_val / step_val,nums_all_correct/nums_all))

if __name__=='__main__':
    main()
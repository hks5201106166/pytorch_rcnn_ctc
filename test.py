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
def config_args():
    parser = argparse.ArgumentParser()
    # the config for the train
    parser.add_argument('--config',default='datasets/config/config.yaml',type=str,help='the path of the images dir')
    args = parser.parse_args()
    with open(args.config,'r') as file:
        config=yaml.load(file,Loader=yaml.FullLoader)
        config=edict(config)
    return config

def test():
    config=config_args()

    transform_val=\
        transforms.Compose([
            transforms.Resize(size=(config.DATASET.IMAGE_SIZE.H,config.DATASET.IMAGE_SIZE.W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

    criterion=nn.CTCLoss()
    label_tool=LabelTool(char_std_path=config.DATASET.CHAR_FILE)

    model=crnn.CRNN(config).to(torch.device("cuda:"+config.CUDNN.GPU))
    model.load_state_dict(torch.load(config.TRAIN.RESUME.MODEL_SAVE)['state_dict'])
    images_val,labels_val=label_tool.get_labels(labels_path=config.DATASET.LABELS_FILE.VAL)
    dataset_val=Dataset_OCR(images_root_dir=config.DATASET.IMAGE_ROOT,images_name=images_val,transform=transform_val)
    dataloader_val=DataLoader(dataset=dataset_val,batch_size=config.TEST.BATCH,shuffle=config.TEST.SHUFFLE,num_workers=config.TEST.WORKERS)




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
    print("val_loss_avarage={},val_accuracy={}".format(loss_all_val / step_val,nums_all_correct/nums_all))


if __name__=='__main__':
    test()
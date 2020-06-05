#-*-coding:utf-8-*-
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
import torch
import logging
from time import gmtime, strftime
import os
def get_logger(save_outputs):
    t = save_outputs
    log_path = 'save_outputs/'+t+'/log/'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logfile = log_path +'log'+ '.txt'
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger
def get_output_dir(blank=''):
    t = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    if not os.path.exists('save_outputs'):
        os.makedirs('save_outputs')
    log_path = 'save_outputs/'
    save_outputs=log_path+t
    os.mkdir(save_outputs+'_'+blank)
    os.mkdir(save_outputs+'_'+blank+'/log')
    os.mkdir(save_outputs+'_'+blank+'/models_saved')
    return t+'_'+blank
class LabelTool(object):
    def __init__(self,char_std_path):
        with open(char_std_path,'rb') as file:
            id_map_str = {}
            str_map_id={}
            for num, char in enumerate(file.readlines()):
                id_map_str[num] = char.decode('utf-8').strip()
                str_map_id[char.decode('utf-8').strip()]=num
        self.id_map_str=id_map_str
        self.str_map_id=str_map_id
    def encode(self):
        pass
    def decode(self,output):
        l=len(output)
        pred_str = []
        pred_str_blank=[]
        for i in range(l):
            if output[i] != 0 and (not (i > 0 and output[i - 1] == output[i])):
                pred_str.append(self.id_map_str[output[i]])
            if output[i] != 0:
                pred_str_blank.append(self.id_map_str[output[i]])
            else:
                pred_str_blank.append('_')

        #for i in range(l):

        #print(pred_str)
        #print(pred_str1)
        return ''.join(pred_str),''.join(pred_str_blank)
    def decode_labels(self,target,target_lengths):
        target=target.numpy()
        target_lengths=target_lengths.numpy().tolist()
        l_start=0
        l_end=0
        targets_str=[]
        for index in range(len(target_lengths)):
            target_str=''
            target_length=target_lengths[index]
            l_end+=target_length
            t=target[l_start:l_end]
            for id in t:
                word=self.id_map_str[id]
                target_str+=word
            l_start+=target_length
            targets_str.append(target_str)
        return targets_str



    def decode_batch(self,outputs):
        pred_strs=[]
        pred_strs_blank=[]
        for output in outputs:
            pred_str,pred_str_blank=self.decode(list(output))
            pred_strs.append(pred_str)
            pred_strs_blank.append(pred_str_blank)
        return pred_strs,pred_strs_blank
    def cal_correct_nums(self,pred_strs,preds_strs_val_blank,indexs,labels_train,step_val):
        indexs = indexs.numpy()
        N = indexs.shape[0]
        labels = labels_train[indexs]
       # print(labels)
        labels_str=labels
        # for label in labels:
        #     label_str,_=self.decode([int(word.strip()) for word in label.split(' ')])
        #     labels_str.append(label_str)
        #print(labels_str)
        correct_nums=0
        for i in range(len(labels_str)):
            #print(labels_str[i])
            if pred_strs[i]==labels_str[i]:
                correct_nums+=1
        return correct_nums

    @staticmethod
    def get_labels(labels_path):
        images_name = []
        labels = []
        print('generator the images_name and labels:')
        with open(labels_path, 'r') as files:
            lines = files.readlines()
            for index in tqdm(range(0, len(lines))):
                file = lines[index]
                image_name = file.split('jpg')[0] + 'jpg'
                label = file.split('jpg')[1].strip()
                images_name.append(image_name)
                labels.append(label)
                # if index%10000==0:
                #     print(index)
            return images_name, np.array(labels)

    def convert_ctcloss_labels(self,indexs, labels_train_val,sequence_len):
        indexs = indexs.numpy()
        N = indexs.shape[0]
        labels = labels_train_val[indexs]
        labels_ = []
        target_lengths = []
        for label in labels:
            for word in list(label):
                labels_.append(self.str_map_id[word])
            target_lengths.append(len(list(label)))

        labels_tensor = torch.tensor(labels_, dtype=torch.int32)
        input_lengths = torch.full(size=(N,), fill_value=sequence_len, dtype=torch.int32)
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32)
        return labels_tensor, input_lengths, target_lengths





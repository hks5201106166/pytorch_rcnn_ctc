#-*-coding:utf-8-*-
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
import torch
import logging
from time import gmtime, strftime
import os
from tqdm import tqdm
from tqdm._tqdm import trange

class Avgloss():
    '''
    the average of the loss
    '''
    def __init__(self):
        self.loss_all=0
        self.step=0
    def print(self):
        average_loss=self.loss_all/self.step
        return average_loss
def get_logger(save_outputs):
    """
    @param save_outputs:the log and model save dir
    @return:
    """
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
    """
    @param blank: make the dir for the save_outputs
    @return: the dir for the save_outputs
    """
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
    """
    the tool for label,including encode,decode,cal_correct_nums .ect
    """
    def __init__(self,char_std_path):
        '''
        @param char_std_path: the char_std dict root
        '''
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
        '''
        @param output: one line words decode
        @return: the str word
        '''
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
        '''

        @param outputs: mutil line words decode
        @return: the mutil line words
        '''
        pred_strs=[]
        for output in outputs:
            pred_str,pred_str_blank=self.decode(list(output))
            pred_strs.append(pred_str)

        return pred_strs
    def cal_correct_nums(self,pred_strs,indexs,labels_train,step_val):
        '''
        calculate the correct nums
        @param pred_strs: the model pred_strs
        @param indexs: the idexs of the labels
        @param labels_train: the ground of the labels

        @return:
        '''
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
            # if step_val==0:
            #     print('{}     {}'.format(labels_str[i],pred_strs[i]))
            #print(labels_str[i])
            if pred_strs[i]==labels_str[i]:
                correct_nums+=1
        return correct_nums

    def get_labels(self,labels_path):
        '''
        get the labels from  .txt
        @param labels_path: the labels .txt file path
        @return:
        '''
        images_name = []
        labels = []
        print('generator the images_name and labels:')
        with open(labels_path, 'r') as files:
            lines = files.readlines()

            for index in tqdm(range(0, len(lines))):
                len_label=0
                file = lines[index]
                image_name = file.split('jpg')[0] + 'jpg'
                label = file.split('jpg')[1].strip()
                for word in list(label):
                    if word in self.str_map_id:
                        len_label += 1
                if len_label!=0:
                    images_name.append(image_name)
                    labels.append(label)
            return images_name, np.array(labels)

    def convert_ctcloss_labels(self,indexs, labels_train_val,sequence_len):
        '''
        convert the labels format for the ctc loss
        @param indexs: the indexs of the labels
        @param labels_train_val: the labels ids for the ctc loss
        @param sequence_len: the len of the labels
        @return: labels_tensor, input_lengths, target_lengths
        '''
        indexs = indexs.numpy()
        N = indexs.shape[0]
        labels = labels_train_val[indexs]
        labels_ = []
        target_lengths = []
        for label in labels:
            len_label=0
            for word in list(label):
                if word in self.str_map_id:
                    labels_.append(self.str_map_id[word])
                    len_label+=1
            target_lengths.append(len_label)
            if len_label==0:
                print('labels is lenght zeros')
                break

        labels_tensor = torch.tensor(labels_, dtype=torch.int32)
        input_lengths = torch.full(size=(N,), fill_value=sequence_len, dtype=torch.int32)
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32)
        return labels_tensor, input_lengths, target_lengths
def train_one_epoch(epoch,dataloader_train,config,model,label_tool, labels_train,criterion,avgloss,optimizer,scheduler,logger):
    '''
    train the model
    @param epoch: the epoch for train
    @param dataloader_train: the dataloader for the train
    @param config:
    @param model:
    @param label_tool:
    @param labels_train:
    @param criterion:
    @param avgloss:
    @param optimizer:
    @param scheduler:
    @param logger:
    @return: none
    '''
    step_epoch = len(dataloader_train)
    for i, (images, indexs) in enumerate(dataloader_train):
        images = images.to(torch.device("cuda:" + config.CUDNN.GPU))

        output = model(images)
        sequence_len = output.shape[0]
        target, input_lengths, target_lengths = label_tool.convert_ctcloss_labels(indexs, labels_train, sequence_len)
        loss = criterion(output.cpu(), target, input_lengths, target_lengths)
        # loss_all+=loss.cpu().detach().numpy()
        avgloss.loss_all += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(scheduler.get_lr())
        avgloss.step += 1
        if avgloss.step % config.TRAIN.SHOW_STEP == 0:
            print("epoch:{},step:({}/{}),loss={:.6f},loss_avarage={:.6f},lr={}".format(epoch, avgloss.step, step_epoch, loss,
                                                                                     avgloss.loss_all / avgloss.step,
                                                                                     scheduler.get_lr()[0]))
            logger.debug(
                "epoch:{},step:{},loss={:.6f},loss_avarage={:.6f},lr={}".format(epoch, avgloss.step, loss, avgloss.loss_all/avgloss.step,
                                                                                scheduler.get_lr()[0]))
def validate(epoch,dataloader_val,labels_val,config,model,label_tool,criterion,save_outputs,logger):
    '''
    validate the model
    @param epoch:
    @param dataloader_val:
    @param labels_val:
    @param config:
    @param model:
    @param label_tool:
    @param criterion:
    @param save_outputs:
    @param logger:
    @return: none
    '''

    loss_all_val = 0
    step_val = 0
    nums_all = 0
    nums_all_correct = 0
    pbar = tqdm(total=100)
    with torch.no_grad():
        for index,(images_val, indexs_val) in enumerate(dataloader_val):
            pbar.update(100/len(dataloader_val))
            images_val = images_val.to(torch.device("cuda:" + config.CUDNN.GPU))

            output_val = model(images_val)
            sequence_len_val = output_val.shape[0]
            target_val, input_lengths_val, target_lengths_val = label_tool.convert_ctcloss_labels(indexs_val,
                                                                                                  labels_val,
                                                                                                  sequence_len_val)
            preds_val = output_val.permute(1, 0, 2).argmax(2).cpu().numpy()
            preds_str_val= label_tool.decode_batch(preds_val)
            correct_nums = label_tool.cal_correct_nums(preds_str_val, indexs_val, labels_val,step_val)
            nums_all_correct += correct_nums
            nums_all += output_val.shape[1]
            #print('nums_all_correct{},nums_all{}'.format(nums_all_correct, nums_all))

            loss_val = criterion(output_val, target_val, input_lengths_val, target_lengths_val)
            loss_all_val += loss_val
            step_val += 1
    pbar.close()
    acc = nums_all_correct / nums_all
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "acc": acc,
        }, os.path.join('save_outputs/' + save_outputs + '/models_saved/', "epoch_{}_acc_{:.4f}.pth".format(epoch, acc))
    )
    print('......................................................................................')
    print("epoch:{},val_loss_avarage={},val_accuracy={}".format(epoch, loss_all_val / step_val,
                                                                nums_all_correct / nums_all))
    print('......................................................................................')
    logger.debug("epoch:{},val_loss_avarage={},val_accuracy={}".format(epoch, loss_all_val / step_val,
                                                                       nums_all_correct / nums_all))





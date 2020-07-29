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
import csv
import cv2
from collections import defaultdict
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
    def cal_correct_nums(self,images_val,pred_strs,indexs):
        '''
        calculate the correct nums
        @param pred_strs: the model pred_strs
        @param indexs: the idexs of the labels
        @param labels_train: the ground of the labels

        @return:
        '''
        # images=images_val.permute(2,3,0,1).squeeze().cpu().numpy()
        # image = np.uint8(((images[:, :,1]+1) * 0.5 * 255))
        # cv2.imshow('ttt',image)
        # cv2.waitKey(0)
        N = len(indexs)
       # labels = labels_train[indexs]
       # print(labels)
        labels_str=indexs
        # for label in labels:
        #     label_str,_=self.decode([int(word.strip()) for word in label.split(' ')])
        #     labels_str.append(label_str)
        #print(labels_str)
        correct_nums=0
        idcard_error=[]
        for index,label_name in enumerate(labels_str):
            label_ground_true=labels[label_name]
            tt=label_ground_true[i]
            if pred_strs[index]==label_ground_true[i]:
                correct_nums+=1
            else:
                idcard_error.append(index)
                image = np.uint8((images[:, :, int(index)] + 1) * 0.5 * 255)
                # print('true:{},pred:{}'.format(tt,pred_strs[index]))
                cv2.imwrite('/home/ubuntu/hks/ocr/pytorch_rcnn_ctc_idcard/pytorch_rcnn_ctc/error_samples/'+'true:{},pred:{}'.format(tt,pred_strs[index])+'.jpg',image)
                # cv2.imshow('hkt', image)
                # cv2.waitKey(0)
        return correct_nums,idcard_error

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
    def get_labels_with_fake_data(self,labels_path,config):
        '''
        get the labels from  .txt
        @param labels_path: the labels .txt file path
        @return:
        '''
        d_labels={}
        for i in range(1,20):
            d_labels[str(i)]=[]

        dirs=os.listdir(labels_path)

        for dir in dirs:
            l=dir.split('.txt')[0]
            labels=open(os.path.join(labels_path,dir),'r').readlines()
            d_labels[l].extend(labels)



        # labels_csv_fake_val=list(csv.reader(open(config.DATASET.LABELS_FILE.VAL_FAKE, 'r', encoding='utf-8-sig')))
        # labels_csv_fake.extend(labels_csv_fake_val)


        return d_labels
    def get_idcard_labels(self,labels_path):
        '''
        get the labels from  .txt
        @param labels_path: the labels .txt file path
        @return:
        '''

        labels=open(labels_path,'r').readlines()
        return labels

    def convert_ctcloss_labels(self,labels,sequence_len):
        '''
        convert the labels format for the ctc loss
        @param indexs: the indexs of the labels
        @param labels_train_val: the labels ids for the ctc loss
        @param sequence_len: the len of the labels
        @return: labels_tensor, input_lengths, target_lengths
        '''
        #indexs = indexs.numpy()
        N = len(labels)
        target_lengths = []
        labels_ = []
        for label in labels:
            texts=label.split(' ')[1:]

            len_label=0

            for word in texts:

               # print(word)
                if word in self.str_map_id:
                    labels_.append(self.str_map_id[word])
                    len_label+=1
                # else:
                #     print('the key {} is not in the dict'.format(word))

            target_lengths.append(len_label)
            if len_label==0:
                print('labels is lenght zeros')
                print(label)
                # break

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
    for ind, (images0,images1,images2,images3,images4,images5,images6,\
               images7,images8,images9,images10,images11,images12,\
               images13,images14,images15,images16,images17,images18,\
               labels_list0,labels_list1,labels_list2,labels_list3,labels_list4 ,\
               labels_list5,labels_list6,labels_list7,labels_list8,labels_list9, \
               labels_list10, labels_list11, labels_list12, labels_list13, labels_list14, \
               labels_list15, labels_list16, labels_list17, labels_list18) in enumerate(dataloader_train):
        #images = images.to(torch.device("cuda:" + config.CUDNN.GPU))

        images=[images0.to(torch.device("cuda:" + config.CUDNN.GPU)),images1.to(torch.device("cuda:" + config.CUDNN.GPU)),
                images2.to(torch.device("cuda:" + config.CUDNN.GPU)),images3.to(torch.device("cuda:" + config.CUDNN.GPU)),
                images4.to(torch.device("cuda:" + config.CUDNN.GPU)),images5.to(torch.device("cuda:" + config.CUDNN.GPU)),
                images6.to(torch.device("cuda:" + config.CUDNN.GPU)),images7.to(torch.device("cuda:" + config.CUDNN.GPU)),
                images8.to(torch.device("cuda:" + config.CUDNN.GPU)),images9.to(torch.device("cuda:" + config.CUDNN.GPU)),
                images10.to(torch.device("cuda:" + config.CUDNN.GPU)),images11.to(torch.device("cuda:" + config.CUDNN.GPU)),
                images12.to(torch.device("cuda:" + config.CUDNN.GPU)),images13.to(torch.device("cuda:" + config.CUDNN.GPU)),
                images14.to(torch.device("cuda:" + config.CUDNN.GPU)),images15.to(torch.device("cuda:" + config.CUDNN.GPU)),
                images16.to(torch.device("cuda:" + config.CUDNN.GPU)),images17.to(torch.device("cuda:" + config.CUDNN.GPU)),
                images18.to(torch.device("cuda:" + config.CUDNN.GPU))]
        labels=[ labels_list0,labels_list1,labels_list2,labels_list3,labels_list4 ,\
               labels_list5,labels_list6,labels_list7,labels_list8,labels_list9, \
               labels_list10, labels_list11, labels_list12, labels_list13, labels_list14, \
               labels_list15, labels_list16, labels_list17, labels_list18]
        #np.random.shuffle(images)
        loss_texts=0
        loss_dict={}
        rexts=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
        for i,image in enumerate(images):
            label=labels[i]
            output = model(image)
            sequence_len = output.shape[0]
            target, input_lengths, target_lengths = label_tool.convert_ctcloss_labels(label,
                                                                                      sequence_len)
            loss = criterion(output.cpu(), target, input_lengths, target_lengths)
            loss_texts+=loss.item()
            loss_dict[rexts[i]]=loss.item()
            # loss_all+=loss.cpu().detach().numpy()
            avgloss.loss_all += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avgloss.step += 1
        # print(scheduler.get_lr())

        if avgloss.step % config.TRAIN.SHOW_STEP == 0:
            print("epoch:{},step:({}/{}),loss={:.6f},loss_avarage={:.6f},"
                  "loss_1={:.6f},loss_2={:.6f},loss_3={:.6f},"
                  "loss_4={:.6f},loss_5={:.6f},loss_6={:.6f},"
                  "loss_7={:.6f},loss_8={:.6f},loss_9={:.6f},"
                  "loss_10={:.6f},loss_11={:.6f},loss_12={:.6f},"
                  "loss_13={:.6f},loss_14={:.6f},loss_15={:.6f},"
                  "loss_16={:.6f},loss_17={:.6f},loss_18={:.6f},"
                  "loss_19={:.6f},lr={}".format(epoch, avgloss.step, step_epoch, loss_texts/len(images),avgloss.loss_all / avgloss.step,
                                                    loss_dict['1'],loss_dict['2'],loss_dict['3'],loss_dict['4'],
                                                    loss_dict['5'],loss_dict['6'],loss_dict['7'],loss_dict['8'],
                                                    loss_dict['9'], loss_dict['10'], loss_dict['11'], loss_dict['12'],
                                                    loss_dict['13'], loss_dict['14'], loss_dict['15'], loss_dict['16'],
                                                    loss_dict['17'],loss_dict['18'],loss_dict['19'],scheduler.get_lr()[0]))
            logger.debug(
                "epoch:{},step:{},loss={:.6f},loss_avarage={:.6f},lr={}".format(epoch, avgloss.step,loss_texts/len(images), avgloss.loss_all/avgloss.step,
                                                                                scheduler.get_lr()[0]))
def train_one_epoch_dizhi_and_xingming(epoch,dataloader_train,config,model,label_tool, labels_train,criterion,avgloss,optimizer,scheduler,logger):
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
    for ind, (xingming_rect,dizhi_rect,xingbie_rect,mingzhu_rect,shengfengzhenghao_rect,chusheng_rect_year,chusheng_rect_month,chusheng_rect_day,qianfajiguang_rect,youxiaoqixian_rect,indexs) in enumerate(dataloader_train):
        #images = images.to(torch.device("cuda:" + config.CUDNN.GPU))
        xingming_rect=xingming_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
        dizhi_rect=dizhi_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
        if epoch<30:
            flag=0
        else:
            flag=np.random.randint(0,10)
        # flag = np.random.randint(0, 10)
        if flag==0:
            xingbie_rect=xingbie_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            mingzhu_rect=mingzhu_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            shengfengzhenghao_rect=shengfengzhenghao_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            chusheng_rect_year=chusheng_rect_year.to(torch.device("cuda:" + config.CUDNN.GPU))
            chusheng_rect_month=chusheng_rect_month.to(torch.device("cuda:" + config.CUDNN.GPU))
            chusheng_rect_day=chusheng_rect_day.to(torch.device("cuda:" + config.CUDNN.GPU))
            #qianfajiguang_rect=qianfajiguang_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            youxiaoqixian_rect=youxiaoqixian_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            qianfajiguang_rect = qianfajiguang_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            images=[xingming_rect,mingzhu_rect,xingbie_rect,chusheng_rect_year,chusheng_rect_month,chusheng_rect_day,dizhi_rect,shengfengzhenghao_rect,qianfajiguang_rect,youxiaoqixian_rect]
            rexts = ['xingming', 'mingzhu', 'xingbie', 'chusheng_year',
                     'chusheng_month', 'chusheng_day', 'dizhi', 'shengfengzhenghao',
                     'qianfajiguang', 'youxiaoqixian']
        else:
            images = [xingming_rect, dizhi_rect]
            rexts=['xingming','dizhi']
            #np.random.shuffle(images)

        if flag==0:
            loss_texts = 0
            loss_dict = {}
            for i,image in enumerate(images):
                output = model(image)
                sequence_len = output.shape[0]
                target, input_lengths, target_lengths = label_tool.convert_ctcloss_labels(indexs, labels_train,
                                                                                          sequence_len, i)

                loss = criterion(output.cpu(), target, input_lengths, target_lengths)
                loss_texts+=loss.item()
                loss_dict[rexts[i]]=loss.item()
                # loss_all+=loss.cpu().detach().numpy()
                avgloss.loss_all += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avgloss.step += 1
            # print(scheduler.get_lr())

            if avgloss.step % config.TRAIN.SHOW_STEP == 0:
                print("epoch:{},step:({}/{}),loss={:.6f},loss_avarage={:.6f},"
                      "loss_xingming={:.6f},loss_mingzhu={:.6f},loss_xingbie={:.6f},"
                      "loss_chusheng_year={:.6f},loss_chusheng_month={:.6f},loss_chusheng_day={:.6f},"
                      "loss_dizhi={:.6f},loss_shengfengzhenghao={:.6f},loss_qianfajiguang={:.6f},"
                      "loss_youxiaoqixian={:.6f},lr={}".format(epoch, avgloss.step, step_epoch, loss_texts/len(images),avgloss.loss_all / avgloss.step,
                                                        loss_dict['xingming'],loss_dict['mingzhu'],loss_dict['xingbie'],loss_dict['chusheng_year'],
                                                        loss_dict['chusheng_month'],loss_dict['chusheng_day'],loss_dict['dizhi'],loss_dict['shengfengzhenghao'],
                                                        loss_dict['qianfajiguang'],loss_dict['youxiaoqixian'],
                                                                                         scheduler.get_lr()[0]))
        else:
            loss_texts = 0
            loss_dict = {}
            for i, image in enumerate(images):
                output = model(image)
                sequence_len = output.shape[0]
                if i == 0:
                    target, input_lengths, target_lengths = label_tool.convert_ctcloss_labels(indexs, labels_train,
                                                                                              sequence_len, 0)
                else :
                    target, input_lengths, target_lengths = label_tool.convert_ctcloss_labels(indexs, labels_train,
                                                                                              sequence_len, 6)
                # else:
                #     target, input_lengths, target_lengths = label_tool.convert_ctcloss_labels(indexs, labels_train,
                #                                                                               sequence_len, 8)
                loss = criterion(output.cpu(), target, input_lengths, target_lengths)
                loss_texts += loss.item()
                loss_dict[rexts[i]] = loss.item()
                # loss_all+=loss.cpu().detach().numpy()
                avgloss.loss_all += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avgloss.step += 1
            # print(scheduler.get_lr())

            if avgloss.step % config.TRAIN.SHOW_STEP == 0:
                print("epoch:{},step:({}/{}),loss={:.6f},loss_avarage={:.6f},"
                      "loss_xingming={:.6f},"
                      "loss_dizhi={:.6f},"

                      .format(epoch, avgloss.step, step_epoch,
                                                               loss_texts / len(images),
                                                               avgloss.loss_all / avgloss.step,
                                                               loss_dict['xingming'],
                                                               loss_dict['dizhi'],

                                                               scheduler.get_lr()[0]))
            # print(scheduler.get_lr())


            logger.debug(
                "epoch:{},step:{},loss={:.6f},loss_avarage={:.6f},lr={}".format(epoch, avgloss.step,loss_texts/len(images), avgloss.loss_all/avgloss.step,
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
                                                                                                  sequence_len_val)
            preds_val = output_val.permute(1, 0, 2).argmax(2).cpu().numpy()
            preds_str_val= label_tool.decode_batch(preds_val)
            ground_true=''.join(indexs_val[0].strip().split(' ')[1:])
            if preds_str_val[0]==ground_true:
                correct_nums = 1
            else:
                correct_nums=0
            nums_all_correct += correct_nums
            nums_all += output_val.shape[1]
            # print('nums_all_correct{},nums_all{}'.format(nums_all_correct, nums_all))

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






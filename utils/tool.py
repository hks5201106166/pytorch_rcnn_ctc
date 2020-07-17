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
    def cal_correct_nums(self,pred_strs,indexs,labels,step_val,i):
        '''
        calculate the correct nums
        @param pred_strs: the model pred_strs
        @param indexs: the idexs of the labels
        @param labels_train: the ground of the labels

        @return:
        '''

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
            # tt=label_ground_true[i]
            if pred_strs[index]==label_ground_true[i]:
                correct_nums+=1
            else:
                idcard_error.append(index)
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
    def get_idcard_labels(self,labels_path):
        '''
        get the labels from  .txt
        @param labels_path: the labels .txt file path
        @return:
        '''

        images_name = []
        labels_dict={}
        print('generator the images_name and labels:')
        labels_csv = list(csv.reader(open(labels_path, 'r', encoding='utf-8-sig')))
        for label in labels_csv:
            images_name.append(label[0])
            labels_dict[label[0]] = label[1:]

        return images_name,labels_dict

    def convert_ctcloss_labels(self,indexs, labels_train_val,sequence_len,i):
        '''
        convert the labels format for the ctc loss
        @param indexs: the indexs of the labels
        @param labels_train_val: the labels ids for the ctc loss
        @param sequence_len: the len of the labels
        @return: labels_tensor, input_lengths, target_lengths
        '''
        #indexs = indexs.numpy()
        N = len(indexs)
        target_lengths = []
        labels_ = []
        for index in indexs:
            label = labels_train_val[index]
            texts=label[i]

            len_label=0

            for word in texts.strip():

               # print(word)
                if word in self.str_map_id:
                    labels_.append(self.str_map_id[word])
                    len_label+=1
                # else:
                #     print('the key {} is not in the dict'.format(word))

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
    for ind, (xingming_rect,dizhi_rect,xingbie_rect,mingzhu_rect,shengfengzhenghao_rect,chusheng_rect_year,chusheng_rect_month,chusheng_rect_day,qianfajiguang_rect,youxiaoqixian_rect,indexs) in enumerate(dataloader_train):
        #images = images.to(torch.device("cuda:" + config.CUDNN.GPU))
        xingming_rect=xingming_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
        dizhi_rect=dizhi_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
        xingbie_rect=xingbie_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
        mingzhu_rect=mingzhu_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
        shengfengzhenghao_rect=shengfengzhenghao_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
        chusheng_rect_year=chusheng_rect_year.to(torch.device("cuda:" + config.CUDNN.GPU))
        chusheng_rect_month=chusheng_rect_month.to(torch.device("cuda:" + config.CUDNN.GPU))
        chusheng_rect_day=chusheng_rect_day.to(torch.device("cuda:" + config.CUDNN.GPU))
        qianfajiguang_rect=qianfajiguang_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
        youxiaoqixian_rect=youxiaoqixian_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
        images=[xingming_rect,mingzhu_rect,xingbie_rect,chusheng_rect_year,chusheng_rect_month,chusheng_rect_day,dizhi_rect,shengfengzhenghao_rect,qianfajiguang_rect,youxiaoqixian_rect]
        #np.random.shuffle(images)
        loss_texts=0
        loss_dict={}
        rexts=['xingming','mingzhu','xingbie','chusheng_year',
               'chusheng_month','chusheng_day','dizhi','shengfengzhenghao',
               'qianfajiguang','youxiaoqixian']
        for i,image in enumerate(images):
            output = model(image)
            sequence_len = output.shape[0]
            target, input_lengths, target_lengths = label_tool.convert_ctcloss_labels(indexs, labels_train,
                                                                                      sequence_len,i)
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

        flag=np.random.randint(0,15)
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
        for i, (xingming_rect, dizhi_rect, xingbie_rect, mingzhu_rect, shengfengzhenghao_rect, chusheng_rect_year,
                chusheng_rect_month, chusheng_rect_day, qianfajiguang_rect, youxiaoqixian_rect, indexs_val) in enumerate(
                dataloader_val):
            pbar.update(100 / len(dataloader_val))
            # images = images.to(torch.device("cuda:" + config.CUDNN.GPU))
            xingming_rect = xingming_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            dizhi_rect = dizhi_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            xingbie_rect = xingbie_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            mingzhu_rect = mingzhu_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            shengfengzhenghao_rect = shengfengzhenghao_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            chusheng_rect_year = chusheng_rect_year.to(torch.device("cuda:" + config.CUDNN.GPU))
            chusheng_rect_month = chusheng_rect_month.to(torch.device("cuda:" + config.CUDNN.GPU))
            chusheng_rect_day = chusheng_rect_day.to(torch.device("cuda:" + config.CUDNN.GPU))
            qianfajiguang_rect = qianfajiguang_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            youxiaoqixian_rect = youxiaoqixian_rect.to(torch.device("cuda:" + config.CUDNN.GPU))
            images = [xingming_rect, mingzhu_rect, xingbie_rect, chusheng_rect_year, chusheng_rect_month,
                      chusheng_rect_day, dizhi_rect, shengfengzhenghao_rect, qianfajiguang_rect, youxiaoqixian_rect]
            # np.random.shuffle(images)
        # for index,(images_val, indexs_val) in enumerate(dataloader_val):
        #     pbar.update(100/len(dataloader_val))
            idcards_error=[]
            for i,images_val in enumerate(images):
           # images_val = images_val.to(torch.device("cuda:" + config.CUDNN.GPU))

                output_val = model(images_val)
                sequence_len_val = output_val.shape[0]
                target_val, input_lengths_val, target_lengths_val = label_tool.convert_ctcloss_labels(indexs_val,
                                                                                                      labels_val,
                                                                                                      sequence_len_val,i)

                preds_val = output_val.permute(1, 0, 2).argmax(2).cpu().numpy()
                preds_str_val= label_tool.decode_batch(preds_val)
                correct_nums,idcard_error= label_tool.cal_correct_nums(preds_str_val, indexs_val, labels_val,step_val,i)
                idcards_error.extend(idcard_error)

                #print('nums_all_correct{},nums_all{}'.format(nums_all_correct, nums_all))
                loss_val = criterion(output_val, target_val, input_lengths_val, target_lengths_val)
                loss_all_val += loss_val
                step_val += 1
            nums_error=len(set(idcards_error))
            nums_correct=output_val.shape[1]-nums_error
            nums_all_correct += nums_correct
            nums_all += output_val.shape[1]

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





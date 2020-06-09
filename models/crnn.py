#-*-coding:utf-8-*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from .cnn.resnet import *
from .cnn.densenet import *
class CRNN(nn.Module):
    '''
    the model of the crnn
    '''
    def __init__(self,config):
        super(CRNN, self).__init__()
        if config.MODEL.BACKBONE   == 'resnet18':
            self.cnn=resnet18(config=config)
        elif config.MODEL.BACKBONE == 'resnet34':
            self.cnn=resnet34(config=config)
        elif config.MODEL.BACKBONE == 'resnet50':
            self.cnn = resnet50(config=config)
        elif config.MODEL.BACKBONE == 'resnet101':
            self.cnn=resnet101(config=config)
        elif config.MODEL.BACKBONE == 'resnet152':
            self.cnn = resnet152(config=config)
        elif config.MODEL.BACKBONE == 'densenet':
            self.cnn = densenet121(config=config)
        self.rnn=nn.LSTM(config.MODEL.LSTM_NUM_HIDDEN,config.MODEL.LSTM_NUM_HIDDEN,num_layers=config.MODEL.LSTM_NUM_LAYER,bidirectional=True)
        self.nclass=config.MODEL.NUM_CLASSES
        self.embeding=nn.Linear(config.MODEL.LSTM_NUM_HIDDEN*2,self.nclass)
    def forward(self, x):
        output=self.cnn(x)
        output=output.permute(2,0,1)
        output,_=self.rnn(output)
        T,B,N=output.shape
        output=output.view(-1,N)
        output=self.embeding(output)
        output=output.view(T,B,-1)
        output=F.log_softmax(output,dim=2)
        return output
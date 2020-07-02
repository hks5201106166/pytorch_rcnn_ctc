#-*-coding:utf-8-*-
import csv
import os
import yaml
from easydict import EasyDict as edict
config=yaml.load(open('../config/config.yaml','r'))
config=edict(config)
labels=list(csv.reader(open('../datasets/labels/generate_labels1.csv', 'r', encoding='UTF-8-sig')))
files_idcard=os.listdir(config.DATASET.IMAGE_ROOT)

train_csv=csv.writer(open(config.DATASET.LABELS_FILE.TRAIN,'w'))
val_csv=csv.writer(open(config.DATASET.LABELS_FILE.VAL,'w'))
labels_dict={}
for label in labels:
    labels_dict[label[0]]=label[1:]
    # train_csv.writerow(label)
for file_idcard in files_idcard:
    label_idcard=labels_dict[file_idcard]
    label_idcard.insert(0, file_idcard)
    train_csv.writerow(label_idcard)

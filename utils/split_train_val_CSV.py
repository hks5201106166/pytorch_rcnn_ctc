#-*-coding:utf-8-*-
import csv
from sklearn.model_selection import train_test_split
import csv
import os
import cv2
tt=os.listdir('/home/simple/mydemo/ocr_project/word_recogization/id_datas/split_text_idcard')
# for t in tt:
#     image_name=os.listdir('/home/simple/mydemo/ocr_project/word_recogization/id_datas/split_text_idcard/'+t)
#     for  index,n in enumerate(image_name):
#         im=cv2.imread('/home/simple/mydemo/ocr_project/word_recogization/id_datas/split_text_idcard/'+t+'/'+n)
#         cv2.imshow(str(index),im)
# cv2.waitKey(0)
labels_all_=list(csv.reader(open('../datasets/labels/generate_labels1.csv', 'r', encoding='UTF-8-sig')))
labels_all=[]
for label in labels_all_:
    if label[0] in tt:
        labels_all.append(label)
labels_train,labels_val=train_test_split(labels_all,test_size=0.05, random_state=42)

train_csv=csv.writer(open('../datasets/labels/generate_labels_train.csv','w'))
val_csv=csv.writer(open('../datasets/labels/generate_labels_val.csv','w'))
labels_dict={}
train_csv.writerows(labels_train)
val_csv.writerows(labels_val)
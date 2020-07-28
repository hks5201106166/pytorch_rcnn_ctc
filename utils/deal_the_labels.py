import os
import collections
from sklearn.model_selection import train_test_split
path='/home/ubuntu/hks/ocr/out/'
# file=open(path+'labels.txt').readlines()
# train_labels,val_labels=train_test_split(file,test_size=0.01)
# writer_train=open('/home/ubuntu/hks/ocr/pytorch_rcnn_ctc/datasets/labels/'+'train.txt','w')
# writer_train.writelines(train_labels)
# writer_val=open('/home/ubuntu/hks/ocr/pytorch_rcnn_ctc/datasets/labels/'+'val.txt','w')
# writer_val.writelines(val_labels)


file=open('/home/ubuntu/hks/ocr/pytorch_rcnn_ctc/datasets/labels/train.txt','r')
d={}
file_writers=[]
for i in range(1,20):
    d[str(i)]=[]
    file_writers.append(open('/home/ubuntu/hks/ocr/pytorch_rcnn_ctc/datasets/labels/labels_570w/'+str(i)+'.txt','w'))

labels_str=''
for index,label_ in enumerate(file):
    label=label_.strip().split(' ')[1:]
    l=len(label)
    labels_str+=''.join(label)
    d[str(l)].append(label_)
    print(index)
for i in range(1,20):
    file_writers[i-1].writelines(d[str(i)])
conuter=collections.Counter(labels_str)
# pass
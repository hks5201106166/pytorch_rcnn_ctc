import os
path='/home/ubuntu/hks/ocr/out/'
file=open(path+'labels.txt')
for label in file:
    label=label.split(' ')
    pass
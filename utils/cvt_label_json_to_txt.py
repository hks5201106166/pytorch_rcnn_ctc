import os
import json
char_std_file=open('/home/ubuntu/hks/ocr/pytorch_rcnn_ctc/datasets/char_map/char_std_6006.txt','w')
char_std_file.write('blank\n')
alphabet=json.load(open('/home/ubuntu/hks/ocr/pytorch_rcnn_ctc/datasets/char_map/char_map.json'))
for word in alphabet:
    writer_word=word+'\n'
    char_std_file.write(writer_word)
char_std_file.close()

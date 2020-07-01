import os
import json
char_std_file=open('/home/simple/mydemo/ocr_project/word_recogization/pytorch_rcnn_ctc/datasets/char_map/char_std_6031.txt','w')
char_std_file.write('blank\n')
alphabet=json.load(open('/home/simple/mydemo/ocr_project/word_recogization/pytorch_rcnn_ctc/datasets/char_map/char_map.json'))
for word in alphabet:
    writer_word=word+'\n'
    char_std_file.write(writer_word)
char_std_file.close()

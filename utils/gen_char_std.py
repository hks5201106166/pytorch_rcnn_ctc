#-*-coding:utf-8-*-
import utils.alphabets as alphabets
alphabet=alphabets.alphabet
char_std_file=open('/home/simple/mydemo/ocr_project/CRNN_MYPROJECT/datasets/char_map/char_std.txt','w')
char_std_file.write('blank\n')
for word in alphabet:
    writer_word=word+'\n'
    char_std_file.write(writer_word)
char_std_file.close()
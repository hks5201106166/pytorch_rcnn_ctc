with open('/home/simple/mydemo/ocr_project/CRNN_MYPROJECT/datasets/char_map/char_std_6736.txt', 'rb') as file:
    char_dict={}
    for num,char in enumerate(file.readlines()):
        char_dict[char.strip().decode('utf-8')]=num
    # char_dict = {num:char.strip().decode('gbk','ignore') for num, char in enumerate(file.readlines())}


# processing output
with open('/home/simple/mydemo/ocr_project/CRNN_MYPROJECT/datasets/labels/test.txt') as file:
    file_test=open('/home/simple/mydemo/ocr_project/CRNN_MYPROJECT/datasets/labels/test_.txt', 'w', encoding='utf-8')
    for index,line in enumerate(file):
        name=line.strip().split(' ')[0]
        labels=line.strip().split(' ')[1]
        label_str=''
        for label in list(labels):
            num=char_dict[label]
            label_str+=str(num)
            label_str+=' '
            if label==" ":
                print()
        label_str=label_str.strip()+'\n'
        line_write=name+' '+label_str
        file_test.write(line_write)
        if index%10000==0:
            print(index)
    file_test.close()

# final output
# with open('test.txt', 'w', encoding='utf-8') as file:
#     [file.write(val+'\n') for val in value_list]
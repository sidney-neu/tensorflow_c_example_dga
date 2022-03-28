#_*_ coding: utf-8 _*_
import os
import sys
import getopt
import codecs
import DGA_process
import numpy as np
char_list=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5',
'6', '7', '8', '9', '-', '.', '_']
srcFilePath='./data/train_total_20200709_part1.txt'		#重新训练模型寻找的路径
extraFilePath='./data/train_total_20200709_part2.txt'	#载入旧模型训练新增数据的路径
testFilePath='./data/test_total_20200709.txt'
resultPath='./data/result.txt'
modelPath='./data/model_temp' 
charList={}
batch_size = 1000
#epochs = 12000
epochs = 20
i = 1



for line in char_list:
    charList[line]=i
    i += 1
max_features = i




x_data_test = []
y_data_test = []
testFile = codecs.open(filename=testFilePath, mode='r', encoding='utf-8', errors='ignore')
testlines = testFile.readlines()
for line in testlines:
    if line.strip('\n').strip('\r').strip(' ') == '':
        continue
    x_data = []
    s = line.split('\t')
    x = str(s[0]).lower()
    y = int(s[1])
    for char in x:
        try:
            x_data.append(charList[char])
        except:
            print ('unexpected char' + ' : '+ char)
            x_data.append(0)
    x_data_test.append(x_data)
    y_data_test.append(y)
x_data_test=np.array(x_data_test)
#y_data_test=np.array(y_data_test)
testFile.close()

'''
if( mode == 1 ):
	DGA_process.dga_train(64, max_features, x_data_sum, y_data_sum, x_data_test, y_data_test, batch_size, epochs, modelPath)
elif( mode == 0 ):
	DGA_process.dga_extra_train(64, max_features, x_data_sum, y_data_sum, x_data_test, y_data_test, batch_size, epochs, modelPath)
else:
	print("ERROR mode .\n")
	exit(-1)
'''



test_res=DGA_process.predict_list(64, x_data_test, batch_size, modelPath, resultPath)
true_result=0
for i in range(0, len(test_res)):
    if int(y_data_test[i]) == int(test_res[i]):
        true_result+=1
print ("curracy:%d",true_result*1.0/len(test_res))


exit(0)

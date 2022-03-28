#_*_ coding: utf-8 _*_
import os
import sys
import getopt
import codecs
import DGA_process
import numpy as np
from configparser import ConfigParser

class dga_class():
    def __init__(self,config_path):
        cfg = ConfigParser()
        cfg.read(config_path)
        self.data_path = cfg.get('path','data_path')
        self.model_path = cfg.get('path','model_path')
        self.mode = cfg.getint('mode','mode')
        self.X_list = []
        self.y_list = []
        self.char_list=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5',
'6', '7', '8', '9', '-', '.', '_']
        self.char_dict={}
        self.max_features = 1
        self.batch_size = 1000
        self.epochs = 4
        self.inpout_dim = 64
    def char_map(self):
        i = 1
        for line in self.char_list:
            self.char_dict[line]=i
            i += 1
        self.max_features = i
    def process(self):
        self.char_map()
        self.data_read()
        if(0 < self.mode): #train
            self.train_model()
        else:
            self.test_model()
    def data_read(self):
        datafile = codecs.open(filename=self.data_path, mode='r', encoding='utf-8', errors='ignore')
        lines = datafile.readlines()
        x_list=[]
        y_list=[]
        for line in lines:
            if line.strip('\n').strip('\r').strip(' ') == '':
                continue
            x_node = []
            s = line.split('\t')
            x = str(s[0]).lower()
            y = int(s[1])
            for char in x:
                try:
                    x_node.append(self.char_dict[char])
                except:
                    print ('unexpected char' + ' : '+ char)
                    x_node.append(0)
            self.X_list.append(x_node)
            self.y_list.append(y)
        datafile.close()
    def train_model(self):
        X = np.array(self.X_list)
        y = np.array(self.y_list)
        if 1 == self.mode :
            DGA_process.dga_extra_train(self.inpout_dim, self.max_features, X, y, self.batch_size, self.epochs, self.model_path)
        elif 2 == self.mode :
            print(X[0])
            print(y[0])
            DGA_process.dga_train(self.inpout_dim, self.max_features, X, y, self.batch_size, self.epochs, self.model_path)
    def test_model(self):
        test_res=DGA_process.predict_list(self.inpout_dim, X, self.batch_size, self.model_path, './result.txt')
        true_result=0
        for i in range(0, len(test_res)):
            if int(y_data_test[i]) == int(test_res[i]):
                true_result+=1
        print ("curracy:%d",true_result*1.0/len(test_res))
def main():
    Dga_class = dga_class('./config.ini')
    Dga_class.process()
if __name__ == '__main__':
    main()

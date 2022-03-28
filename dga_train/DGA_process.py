# _*_ coding: utf-8 _*_

import os
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Input, Flatten
from keras.preprocessing import sequence
from keras.layers.core import *
from keras.models import *
from keras.callbacks import *
from keras import optimizers
import numpy as np

checkpoint_file = '../model/model_temp.h5'

def model_design(shape, max_features): #shape表示序列长度，max_feature表示取值范围
    inputs= Input(shape=(shape,))#输入的特征维度
    embedded_input = Embedding(max_features, 128, input_length=shape)(inputs) #特征维度max_features->128
    lstm_out = LSTM(128, return_sequences=True)(embedded_input)
    nn_input = Permute((2, 1))(lstm_out)
    nn_input = Dense(30, activation='relu')(nn_input)
    nn_input = Dense(max_features, activation='softmax')(nn_input)
    nn_output = Permute((2, 1))(nn_input)
    output_1d = Flatten()(nn_output)
    output_1d = Dropout(0.5)(output_1d)
    output_1d = Dense(1)(output_1d)
    outputs = Activation('sigmoid')(output_1d)
    model = Model(input=[inputs], output=outputs)
    return model

#def dga_train(max_len, max_features, X_train, y_train, x_test, y_test, batch_size, epochs, modelPath):
def dga_train(max_len, max_features, X_train, y_train, batch_size, epochs, modelPath):
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    #x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)	#当损失函数不在下降则提前终止
    epoch_save = ModelCheckpoint(checkpoint_file,monitor='val_acc',verbose=1,save_best_only=False, save_weights_only=False, mode='auto', period=1) #每一个epoch都保存一次模型
    history = History()

    model = model_design(max_len, max_features)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.000001)	#优化器，其中lr是学习率，decay是学习率衰减的值
    opt = optimizers.RMSprop(lr=0.001)	#优化器，其中lr是学习率，decay是学习率衰减的值
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stopping,history])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_split=0.1, callbacks=[early_stopping,history,epoch_save])
    save_model(model,modelPath+'.h5')
    #model.save(modelPath)

#def dga_train(max_len, max_features, X_train, y_train, x_test, y_test, batch_size, epochs, modelPath):
def dga_extra_train(max_len, max_features, X_train, y_train, batch_size, epochs, modelPath):
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    #x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    epoch_save = ModelCheckpoint(checkpoint_file,monitor='val_acc',verbose=1,save_best_only=False, save_weights_only=False, mode='auto', period=1) 
    history = History()
    model = load_model(modelPath+'.h5')
    #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stopping,history])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_split=0.1, callbacks=[early_stopping,history,epoch_save])
    save_model(model,modelPath+'.h5')


def predict_list(max_len, X_test, batch_size, modelPath, resultPath):
    test_result = []
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)
    my_model = load_model(modelPath+'.h5')
    y_test = my_model.predict(X_test, batch_size=batch_size).tolist()

    file = open(resultPath, 'w+')
    for index in y_test:
        y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
        file.write(str(y)+'\n')
        if y > 0.5:
            aa = 1
        else:
            aa = 0
        test_result.append(aa)
    return test_result

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:45:50 2017

@author: jacob
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import pandas as pd
#import tensorflow as tf
import tflearn
from tflearn.data_utils import load_csv, pad_sequences

def split_training_testing(X, y):
    # split the data into training and testing for hold-out testing    
    n_rows, n_features = np.shape(X)
        
    train_size = int(n_rows*0.7)
    validation_size = int(n_rows*0.8)
       
    X_train = X[0:train_size]
    y_train = y[0:train_size, :]

    X_validation = X[train_size:validation_size]
    y_validation = y[train_size:validation_size, :]
        
    X_test = X[validation_size:n_rows]
    y_test = y[validation_size:n_rows, :]
          
    return (X_train, y_train, X_validation, y_validation, X_test, y_test)

def predict_as_txt(mem_ids, model, label, padLen):
    df1 = pd.read_csv('/Users/jacob/Desktop/Python/Guangxi Market/clean_data1.txt', sep=',', header=0)
    df2 = pd.read_csv('/Users/jacob/Desktop/Python/Guangxi Market/clean_data2.txt', sep=',', header=0)
    df = pd.concat([df1, df2])
    map_df = df[['brand', 'map']].drop_duplicates()
    lines = []
    for id in mem_ids:
        temp = df.loc[df['mem_id']==int(id), ['map']]
        if temp.shape[0] == 0:
            lines.append(id + ':N/A')
        else:
            input = np.array(temp['map'])
            if len(input) < padLen:
                z = np.zeros(padLen - len(input))
                input = np.concatenate((z, input), axis=0).astype(int)
            else:
                input = input[-padLen:]
            input = input[np.newaxis]
            r = model.predict_label(input)
            p = model.predict(input)
            p[0].sort()
            top5 = r[:5]
            top5p = p[0][-5::-1]
            s = id + ':'
            for i in range(0,5):
                tb = map_df.loc[map_df['map']==top5[0][i], 'brand']
                s = s + '(' + tb.values[0] + ',' + str(top5p[i]) + ');'
            lines.append(s.rstrip(';'))

    with open('/Users/jacob/Desktop/Python/Guangxi Market/' + label + '_results_s' + str(padLen) + '.txt', 'w') as f:
        for line in lines:
            f.write(line + '\n')

def main():
    padLen = 10
    X, y = load_csv('/Users/jacob/Desktop/Python/Guangxi Market/job_data.txt',\
                    target_column=-1, has_header=False, categorical_labels=True, n_classes=3817)
    # trainX, trainY, validX, validY, testX, testY = split_training_testing(X, y)

    # Data preprocessing
    # trainX = np.array(trainX).astype('int32')
    # validX = np.array(validX).astype('int32')
    # testX = np.array(testX).astype('int32')
    X = pad_sequences(X, maxlen=padLen, padding='pre', truncating='pre', dtype='int32')

    # Network building
    net = tflearn.input_data([None, padLen])  # 3817*5
    net = tflearn.embedding(net, input_dim=3817, output_dim=256)   # input_dim = 2204898
    net = tflearn.lstm(net, 256, dropout=0.7)
    net = tflearn.fully_connected(net, 3817, activation='softmax')
    m = tflearn.optimizers.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
    net = tflearn.regression(net, optimizer=m,\
                             loss='categorical_crossentropy', batch_size=256)

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    # model.fit(X, y, show_metric=True, n_epoch=10)#, validation_set=(validX, validY))

    # print(model.evaluate(testX, testY))

    # model.save('/Users/jacob/Downloads/job_s' + str(padLen) + '.tflearn')
    model.load('/Users/jacob/Desktop/Python/Guangxi Market/models/job_s' + str(padLen) + '.tflearn')

    off_mem_ids = []
    on_mem_ids = []
    with open('/Users/jacob/Desktop/Python/Guangxi Market/original_offline.txt') as f1,\
     open('/Users/jacob/Desktop/Python/Guangxi Market/original_online.txt') as f2:
     off_lines = f1.readlines()[1:]
     for line in off_lines:
         id = line.split(',', 1)[1].split(',', 1)[0]
         off_mem_ids.append(id)
     on_lines = f2.readlines()[1:]
     for line in on_lines:
         id = line.split(',', 1)[1].split(',', 1)[0]
         on_mem_ids.append(id)
         
    mem_ids = []
    with open('/Users/jacob/Downloads/preTreatmentUsers.txt') as f1:
     lines = f1.readlines()
     for line in lines:
         mem_ids.append(line.rstrip('\n'))

    predict_as_txt(off_mem_ids, model, 'offline', padLen)
    predict_as_txt(on_mem_ids, model, 'online', padLen)

if __name__ == '__main__':
    main()
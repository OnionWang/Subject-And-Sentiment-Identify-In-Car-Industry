# -*- coding: utf-8 -*-

import os
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import util
import models

SEED = 2018

# is_training = sys.argv[1]
is_training = True

param = dict()
param['train_sentiment_path'] = 'dataSet/train_sentiment.csv'
param['test_local_path'] = 'dataSet/test_local.csv'
param['output_path'] = 'result/test_result.csv'
param['cache_dir_path'] = 'cache/'
param['sentiment_ckp_path'] = param['cache_dir_path'] + 'sentiment-ckp/'
if not os.path.exists(param['sentiment_ckp_path']):
    os.makedirs(param['sentiment_ckp_path'])
param['dic_path'] = 'docs/dic.txt'
param['stopWords_path'] = 'docs/stopWords.txt'
param['fasttext_path'] = 'docs/embedding_all_fasttext2_300.txt'
param['merge_path'] = 'docs/embedding_all_merge_300.txt'
param['tencent_path'] = 'docs/embedding_all_tencent_200.txt'
param['sentence_len'] = 60
param['embed_size'] = 300
param['is_training'] = is_training
if not param['is_training']:
    param['test_local_path'] = 'result/subject_result.csv'


DL = ['AT_LSTM', 'GACE', 'HEAT']

X_train, train_sub, y_train, X_test, test_sub, test_id, test_content, test_subject, word2index, sub2index, index2label = util.load_sentiment(param)
param['vocab_size'] = len(word2index) + 1
param['num_class'] = len(index2label)
param['num_subject'] = len(sub2index) + 1

fold = 5
batch_size = 64
epochs = 5

y_target = np.zeros((y_train.shape[0], len(DL), y_train.shape[1]))
y_test_pred = np.zeros((X_test.shape[0], len(DL)*fold, y_train.shape[1]))
n = 0
for name in DL:
    seed = SEED * (n + 1)
    kfold = list(KFold(n_splits = fold, random_state = seed, shuffle = True).split(X_train, y_train)) 
    for i, (train_index, val_index) in enumerate(kfold):
        X, X_sub, y, val_X, val_X_sub, val_y = X_train[train_index], train_sub[train_index], y_train[train_index], X_train[val_index], train_sub[val_index], y_train[val_index]
        util.seed_everything(seed + i)
        model = models.getModel(param, name)
        if i == 0: print(model.summary())
        filepath = param['sentiment_ckp_path'] + name + '-' + str(i + 1)
        if not os.path.exists(filepath):
            checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 2, save_best_only = True, mode = 'min')
            reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 1, min_lr = 0.0001, verbose = 2)
            model.fit([X, X_sub], [y], batch_size, epochs, validation_data = [[val_X, val_X_sub], [val_y]], verbose = 1, callbacks = [reduce_lr, checkpoint])
        model.load_weights(filepath)
        y_pred = np.squeeze(model.predict([val_X, val_X_sub], verbose = 0))
        y_target[val_index, n] = y_pred
        y_test_pred[:, n*fold + i] = np.squeeze(model.predict([X_test, test_sub], verbose = 0))
        f1 = f1_score(val_y.argmax(axis = 1), y_pred.argmax(axis = 1), average = 'macro')
        print('f1 score:%.6f' %f1)
    n += 1

y_target_pred = y_target.mean(axis = 1)
f1 = f1_score(y_train.argmax(axis = 1), y_target.argmax(axis = 1), average = 'macro')
print('Final f1 score:%.4f' %f1)

test_label = y_test_pred.mean(axis = 1)
util.output_sentiment(test_id, test_content, test_subject, test_label.argmax(axis = 1), index2label, param['output_path'])
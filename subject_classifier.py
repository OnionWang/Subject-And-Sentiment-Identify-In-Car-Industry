# -*- coding: utf-8 -*-

import os
import numpy as np

from sklearn.model_selection import KFold
from sklearn.externals import joblib

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import util
import models

SEED = 2018

param = dict()
param['train_subject_path'] = 'dataSet/train_subject.csv'
param['test_subject_path'] = 'dataSet/test_subject.csv'
param['output_path'] = 'result/subject_result.csv'
param['cache_dir_path'] = 'cache/'
param['subject_ckp_path'] = param['cache_dir_path'] + 'subject-ckp/'
if not os.path.exists(param['subject_ckp_path']):
    os.makedirs(param['subject_ckp_path'])
param['dic_path'] = 'docs/dic.txt'
param['stopWords_path'] = 'docs/stopWords.txt'
param['fasttext_path'] = 'docs/embedding_all_fasttext2_300.txt'
param['merge_path'] = 'docs/embedding_all_merge_300.txt'
param['tencent_path'] = 'docs/embedding_all_tencent_200.txt'
param['sentence_len'] = 60
param['embed_size'] = 300

DL = ['TextCnn', 'TextRnn', 'TextRCnn']
ML = ['LogisticRegression', 'SVM', 'RandomForest', 'AdaBoost', 'GradientBoosting']
ALL = DL + ML

X_train, y_train, X_test, test_id, test_content, vocab_size, index2label = util.load_subject(param)
param['vocab_size'] = vocab_size + 1
param['num_class'] = len(index2label) 

fold = 5
batch_size = 64
epochs = 6
y_target = np.zeros((y_train.shape[0], len(ALL), y_train.shape[1]))
y_test_pred = np.zeros((X_test.shape[0], len(ALL)*fold, y_train.shape[1]))
n = 0
for name in DL:
    # print('='*80)
    seed = SEED * (n + 1)
    kfold = list(KFold(n_splits = fold, random_state = seed, shuffle = True).split(X_train, y_train)) 
    for i, (train_index, val_index) in enumerate(kfold):
        X, y, val_X, val_y = X_train[train_index], y_train[train_index], X_train[val_index], y_train[val_index]
        util.seed_everything(seed + i)
        model = models.getModel(param, name)
        # if i == 0: print(model.summary())
        filepath = param['subject_ckp_path'] + name + '-' + str(i + 1)
        if not os.path.exists(filepath):
            reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 1, min_lr = 0.0001, verbose = 2)
            checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 2, save_best_only = True, mode = 'min')
            model.fit(X, y, batch_size, epochs, validation_data = (val_X, val_y), verbose = 2, callbacks = [reduce_lr, checkpoint])
        model.load_weights(filepath)
        y_pred = np.squeeze(model.predict([val_X], verbose = 0))
        y_target[val_index, n] = y_pred
        y_test_pred[:, n*fold + i] = np.squeeze(model.predict([X_test], verbose = 0))
        # threshold = util.search_threshold(np.squeeze(val_y), np.squeeze(y_pred))
        # pre_label = util.get_label_pre(y_pred, threshold)
        # val_label = util.get_label_real(val_y)
        # f1 = util.compute_f1_subject(val_label, pre_label)
        # print('f1 score:%.6f' %f1)
    n += 1

X_train, X_test = util.BagOfWords(param)
for name in ML:
    # print('='*80)
    seed = SEED * (n + 8)
    kfold = list(KFold(n_splits = fold, random_state = seed, shuffle = True).split(X_train, y_train)) 
    for i, (train_index, val_index) in enumerate(kfold):
        X, y, val_X, val_y = X_train[train_index], y_train[train_index], X_train[val_index], y_train[val_index]
        model = models.getModel(param, name)
        filepath = param['subject_ckp_path'] + name + '-' + str(i + 1)
        if not os.path.exists(filepath):
            model.fit(X, y)
            joblib.dump(model, filepath)
        model = joblib.load(filepath)
        y_pred = model.predict_proba(val_X)
        y_target[val_index, n] = y_pred
        y_test_pred[:, n*fold + i] = model.predict_proba(X_test)
        # threshold = util.search_threshold(np.squeeze(val_y), np.squeeze(y_pred))
        # pre_label = util.get_label_pre(y_pred, threshold)
        # val_label = util.get_label_real(val_y)
        # f1 = util.compute_f1_subject(val_label, pre_label)
        # print('f1 score:%.6f' %f1)
    n += 1

y_target_pred = y_target.mean(axis = 1)
threshold = util.search_threshold(y_train, y_target_pred)
pre_label = util.get_label_pre(y_target_pred, threshold)
val_label = util.get_label_real(y_train)
f1 = util.compute_f1_subject(val_label, pre_label)
print('Final threshold:', threshold)
print('Final f1 score:%.6f' %f1)

test_logits = np.squeeze(y_test_pred.mean(axis = 1))
test_label = util.get_label_pre(test_logits, threshold)
util.output_subject(test_id, test_content, test_label, index2label, param['output_path'])
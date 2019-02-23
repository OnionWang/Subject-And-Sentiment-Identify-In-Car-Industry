# -*- coding: utf-8 -*-
import os
import pickle
import jieba
import numpy as np
import pandas as pd
import operator
import tensorflow as tf

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

SEED = 2018

def seed_everything(seed = SEED):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    tf.set_random_seed(SEED)

    
def cut(X):
    seg = jieba.cut(X, cut_all = False)
    return ' '.join(seg)

def clean_puncts(X):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '。', '!', '！']
    for punct in puncts:
        X = X.replace(punct, f' {punct} ')   
    return X

def clean_stopWords(X, stopWords):
    return [x for x in X if x not in stopWords]

def load_subject(param):
    data_pik_path = param['cache_dir_path'] + 'subject_data.pik'
    if os.path.exists(data_pik_path):
        with open(data_pik_path, 'rb') as f:
            return pickle.load(f)

    train_data = pd.read_csv(param['train_subject_path'])
    train_data = train_data.sample(frac = 1, random_state = SEED)
    train_data = train_data.reset_index(drop = True)
    test_data = pd.read_csv(param['test_subject_path'])
    
    X_train = train_data.content
    test_content = test_data.content

    jieba.load_userdict(param['dic_path'])

    X_train = X_train.apply(lambda x : cut(x)).apply(lambda x : x.split())
    X_test = test_content.apply(lambda x : cut(x)).apply(lambda x : x.split())

    with open(param['stopWords_path'], 'r', encoding = 'utf-8') as f:
        stopWords = [line.strip() for line in f.readlines()]

    X_train = X_train.apply(lambda x : clean_stopWords(x, stopWords))
    X_test = X_test.apply(lambda x : clean_stopWords(x, stopWords))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(pd.concat([X_train, X_test]))

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train, maxlen = param['sentence_len'])
    X_test = pad_sequences(X_test, maxlen = param['sentence_len'])

    y_train = train_data.subject
    y_train = y_train.apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    test_id = test_data.content_id
    
    with open(data_pik_path, 'wb') as f:
        pickle.dump((X_train, y_train, X_test, test_id, test_content, len(tokenizer.word_index), mlb.classes_), f)

    return X_train, y_train, X_test, test_id, test_content, len(tokenizer.word_index), mlb.classes_


def BagOfWords(param):
    data_pik_path = param['cache_dir_path'] + 'subject_data2.pik'
    if os.path.exists(data_pik_path):
        with open(data_pik_path, 'rb') as f:
            return pickle.load(f)
    with open(param['stopWords_path'], 'r', encoding = 'utf-8') as f:
        stopWords = [line.strip() for line in f.readlines()]

    train_data = pd.read_csv(param['train_subject_path'])
    train_data = train_data.sample(frac = 1, random_state = SEED)
    train_data = train_data.reset_index(drop = True)
    test_data = pd.read_csv(param['test_subject_path'])

    X_train = train_data.content
    X_test = test_data.content
    jieba.load_userdict(param['dic_path'])
    X_train = X_train.apply(lambda x : cut(x))
    X_test = X_test.apply(lambda x : cut(x))

    tf1 = TfidfVectorizer(ngram_range = (1, 1), analyzer = 'word', stop_words = stopWords, token_pattern = u'[\u4e00-\u9fa5]+', use_idf = True, smooth_idf = True, sublinear_tf = True)
    tf2 = TfidfVectorizer(ngram_range = (1, 2), analyzer = 'char', use_idf = True, smooth_idf = True, sublinear_tf = True)
    data = pd.concat([X_train, X_test], axis = 0)
    w1 = tf1.fit_transform(data)
    w2 = tf2.fit_transform(data)
    w = hstack((w1, w2)).tocsr()

    with open(data_pik_path, 'wb') as f:
        pickle.dump((w[:len(X_train), :], w[len(X_train):, :]), f)

    return w[:len(X_train)], w[len(X_train):]


def load_sentiment(param):
    if param['is_training']:
        data_pik_path = param['cache_dir_path'] + 'sentiment_data.pik'
        if os.path.exists(data_pik_path):
            with open(data_pik_path, 'rb') as f:
                return pickle.load(f)

    train_data = pd.read_csv(param['train_sentiment_path'])
    train_data = train_data.sample(frac = 1, random_state = SEED)
    train_data = train_data.reset_index(drop = True)
    test_data = pd.read_csv(param['test_local_path'])
    
    X_train = train_data.content
    test_content = test_data.content

    jieba.load_userdict(param['dic_path'])

    X_train = X_train.apply(lambda x : cut(x)).apply(lambda x : x.split())
    X_test = test_content.apply(lambda x : cut(x)).apply(lambda x : x.split())

    train_subject = train_data.subject
    test_subject = test_data.subject

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(pd.concat([X_train, X_test, train_subject]))

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train, maxlen = param['sentence_len'])
    X_test = pad_sequences(X_test, maxlen = param['sentence_len'])

    tokenizer2 = Tokenizer()
    tokenizer2.fit_on_texts(train_subject)
    train_subject = tokenizer2.texts_to_sequences(train_subject)
    test_subject = tokenizer2.texts_to_sequences(test_subject)
   
    y_train = train_data.sentiment_value.values

    label2index = {-1:0, 0:1, 1:2}
    index2label = {0:-1, 1:0, 2:1}

    def convert(x):
        tem = np.zeros(3, dtype = int)
        tem[label2index[x]] = 1
        return tem

    y_target = list()
    for i in range(len(y_train)):
        y_target.append(convert(y_train[i]))

    y_train = np.array(y_target)
    test_id = test_data.content_id
    
    if param['is_training']:
        with open(data_pik_path, 'wb') as f:
            pickle.dump((X_train, np.array(train_subject), y_train, X_test, np.array(test_subject), test_id, test_content, test_data.subject, tokenizer.word_index, tokenizer2.word_index, index2label), f)

    return X_train, np.array(train_subject), y_train, X_test, np.array(test_subject), test_id, test_content, test_data.subject, tokenizer.word_index, tokenizer2.word_index, index2label


def load_embedding(path, vocab):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype = 'float32')
    vocab_size = len(vocab) + 1
    embedding_index = dict(get_coefs(*o.split(' ')) for o in open(path, encoding = 'utf-8', errors = 'ignore') if len(o) > 100)
    all_embs = np.stack(embedding_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    np.random.seed(SEED)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, embed_size))
    for word, i in vocab.items():
        if i >= vocab_size: continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key = operator.itemgetter(1))[::-1]

    return unknown_words


def search_threshold(val_y, pre_y):
    y_val = np.transpose(val_y, [1, 0])
    y_pre = np.transpose(pre_y, [1, 0])
    thresholds = list()
    for y_true, y_pred in zip(y_val, y_pre):
        threshold = list()
        for thres in np.arange(0.01, 0.501, 0.01):
            thres = np.round(thres, 2)
            res = f1_score(y_true, (y_pred > thres).astype(int))
            threshold.append([thres, res])
        threshold.sort(key = lambda x: x[1], reverse = True)
        thresholds.append(threshold[0][0])

    return thresholds


def output_subject(test_id, test_content, pre_label, vocabulary_index2label, output_path):
    sub_pre = pd.DataFrame(columns = ['content_id', 'content', 'subject'])
    index = 0
    for content_id, content, sub in zip(test_id, test_content, pre_label):
        for s in sub:
            sub_pre = sub_pre.append(pd.DataFrame({'content_id':content_id, 'content':content, 'subject':vocabulary_index2label[s]}, index = [index]))
            index += 1
    
    sub_pre.to_csv(output_path, index = False)


def output_sentiment(test_id, test_content, test_subject, pre_label, vocabulary_index2label, output_path):
    pre = pd.DataFrame(columns = ['content_id', 'content', 'subject', 'sentiment_value'])
    for index, (content_id, content, subject, sentiment_value) in enumerate(zip(test_id, test_content, test_subject, pre_label)):
        pre = pre.append(pd.DataFrame({'content_id':content_id, 'content':content, 'subject':subject, 'sentiment_value':vocabulary_index2label[sentiment_value]}, index = [index]))

    pre.to_csv(output_path, index = False)


def get_label_pre(logits, threshold):
    pre_list = list()
    for pre in logits:
        index_list = [i for i in range(len(pre)) if pre[i] >= threshold[i]]
        if(not index_list):
            index_list = [np.argmax(pre)]
        pre_list.append(index_list)

    return pre_list


def get_label_real(logits):
    real_list = list()
    for real in logits:
        index_list = [index for index in range(len(real)) if real[index] > 0]
        real_list.append(index_list)

    return real_list


def compute_f1_subject(real, pre, small_value = 0.00001):
    Tp, Fp, Fn = 0, 0, 0
    for i in range(len(real)):
        label = list(set(pre[i] + real[i]))
        for lab in label:
            if lab in real[i] and lab in pre[i]:
                Tp += 1
                pre[i].remove(lab)
                real[i].remove(lab)               
        pred_len = len(pre[i])
        test_len = len(real[i])
        Fp += pred_len
        if pred_len < test_len:
            Fn += test_len - pred_len
    precison = Tp / (Tp + Fp + small_value)
    recall = Tp / (Tp + Fn + small_value)
    F1 = 2 * precison * recall / (precison + recall + small_value)

    return F1


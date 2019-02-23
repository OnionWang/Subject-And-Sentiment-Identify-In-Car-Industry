import pandas as pd
import numpy as np
import tokenization

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

SEED = 2018

def load_subject(param):
    train_data = pd.read_csv(param['train_subject_path'])
    train_data = train_data.sample(frac = 1, random_state = SEED)
    train_data = train_data.reset_index(drop = True)
    test_data = pd.read_csv(param['test_subject_path'])

    data = pd.concat([train_data, test_data])

    tokenizer = tokenization.FullTokenizer(vocab_file = param['vocab_file']) # cut sentence to single word

    ids, masks, segment_ids = list(), list(), list()
    for d in data.content:
        single_input_id, single_input_mask, single_segment_id = convert_single_example(param['sentence_len'], tokenizer, d)
        ids.append(single_input_id)
        masks.append(single_input_mask)
        segment_ids.append(single_segment_id)

    ids = np.asarray(ids, dtype = np.int32)
    masks = np.asarray(masks, dtype = np.int32)
    segment_ids = np.asarray(segment_ids, dtype = np.int32)

    ids_train, ids_test = ids[:len(train_data)], ids[len(train_data):]
    masks_train, masks_test = masks[:len(train_data)], masks[len(train_data):]
    segment_ids_train, segment_ids_test = segment_ids[:len(train_data)], segment_ids[len(train_data):]


    y_train = train_data.subject
    y_train = y_train.apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    test_id = test_data.content_id
    test_content = test_data.content

    return ids_train, masks_train, segment_ids_train, y_train, ids_test, masks_test, segment_ids_test, test_id, test_content, mlb.classes_


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    # 将两个句子相加，如果长度大于max_length 就pop 直到小于 max_length
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_single_example(max_seq_length, tokenizer,text_a,text_b=None):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)# 这里主要是将中文分字
    if tokens_b:
        # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
        # 因为要为句子补上[CLS], [SEP], [SEP]
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 3
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
    # (a) 两个句子:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) 单个句子:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # 这里 "type_ids" 主要用于区分第一个第二个句子。
    # 第一个句子为0，第二个句子是1。在余训练的时候会添加到单词的的向量中，但这个不是必须的
    # 英文[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将中文转换成ids
    # 创建mask
    input_mask = [1] * len(input_ids)
    # 对于输入进行补0
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids


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
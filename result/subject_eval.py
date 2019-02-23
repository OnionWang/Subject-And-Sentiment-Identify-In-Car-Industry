import pandas as pd

test_data = pd.read_csv('dataSet/test_subject.csv')
pred_data = pd.read_csv('result/subject_result.csv')

test = dict()
for i in range(len(test_data)):
    content_id = test_data.iloc[i]['content_id']
    subject = test_data.iloc[i]['subject']
    test[content_id] = subject.split(',')

pred = dict()
for i in range(len(pred_data)):
    content_id = pred_data.iloc[i]['content_id']
    subject = pred_data.iloc[i]['subject']
    if(content_id not in pred):
        pred[content_id] = [subject]
    else:
        pred[content_id].append(subject)

Tp, Fp, Fn = 0, 0, 0
for idx in test:
    label = list(set(pred[idx] + test[idx]))
    for lab in label:
        if lab in test[idx] and lab in pred[idx]:
            Tp += 1
            pred[idx].remove(lab)
            test[idx].remove(lab)
    pred_len = len(pred[idx])
    test_len = len(test[idx])
    Fp += pred_len
    if pred_len < test_len:
        Fn += test_len - pred_len
            
small_value = 0.00001
precison = Tp / (Tp + Fp + small_value)
recall = Tp / (Tp + Fn + small_value)
F1 = 2 * precison * recall / (precison + recall + small_value)
print('正解: %d, 错解: %d, 漏解: %d' %(Tp, Fp, Fn))
print('F1-score: %f' %F1)

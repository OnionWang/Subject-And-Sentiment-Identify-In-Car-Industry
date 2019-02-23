Subject And Sentiment Identify For Users' Opinion In Car Industry
=================

When I first join this competition, I was a beginner for Machine Learning and have no idea about Deep Learning. Thus I finally got 117/1709 rank. I just used traditional Machine Learning way to solve this NLP problem(SVM + LogitsticRegression). After half of year learning, I decide to rewrite this compotition. In order to evaluate my result, I finally fine-tune BERT in this dataSet and compare with my result.

I will tell you how to run my code to reproduce my result, for more details, please read solution.pdf.
By the way, my code is easy style for beginner (run and read).

Thanks!


Enviroment
---------
Python3.x (sklearn, jieba)
Keras
Tensorflow1.X


Code Framework
---------
* bert/: fine-tune bert code
* cache/: save model checkpoint and pre-process data
* code/: 
	* models.py: used models(Fasttext, TextCnn, TextRnn, TextRCnn, ATAE_LSTM, GCAE, HEAT)
	* subject/sentiment_classifier.py: main for subject/sentiment classify
	* util.py: data preprocess and some utils 
* dataSet/:
	* original_data/: original data in this competition(train and test)
	* split_data.py: split data for local subject/sentiment train and test(just for evaluate)
	* train_subject/sentiment.csv
	* test_subject/local.csv
* docs/:
	* dic.txt: subject word in original data(for split words)
	* embedding_all_*.txt: pre-train word embedding(no use)
	* stopWords.txt
* result/:
	* subject/test_eval.py: eval for local test


Usage
--------
**split data to trainset and testset**

cd dataSet/

python split_data.py

subject predict

python code/subject_classifier.py

it will produce subject_result.csv in result/, then you can run subject_eval.py to evaluate the result

**sentiment predict**

train:

python code/sentiment_classifier.py True

test:

python code/sentiment_classifier.py Flase

it will produce test_result.csv in result/, then you can run test_eval.py to evaluate the result


BERT fine-tune
----------
pre-trained model https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
waiting ...

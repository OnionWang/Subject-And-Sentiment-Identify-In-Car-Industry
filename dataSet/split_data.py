import pandas as pd

from sklearn.model_selection import train_test_split


train_data = pd.DataFrame(pd.read_csv('dataSet/original_data/train.csv'))
ds_train = pd.DataFrame(columns = ['content_id', 'content', 'subject', 'sentiment_value'])
dic = dict()
for i in range(len(train_data)):
	content_id = train_data.iloc[i]['content_id']
	content = train_data.iloc[i]['content']
	subject = train_data.iloc[i]['subject']
	sentiment_value = train_data.iloc[i]['sentiment_value']
	if(content not in dic):
		dic[content] = [content_id, subject, str(sentiment_value)]      
	else:
		dic[content][1] += ',' + subject
		dic[content][2] += ',' + str(sentiment_value)

index = 0   
for key, value in dic.items():
	ds_train = ds_train.append(pd.DataFrame({'content_id':value[0], 'content':key, 'subject':value[1], 'sentiment_value':value[2]},index = [index]))
	index += 1

_, _, train, test = train_test_split(ds_train['content'], ds_train, test_size = 0.3, random_state = 0)

train_sub = pd.DataFrame(columns = ['content_id', 'content', 'subject'])
train_sub.content_id = train.content_id
train_sub.content = train.content
train_sub.subject = train.subject
train_sub.to_csv('dataSet/train_subject.csv',index = False)

test_sub = pd.DataFrame(columns = ['content_id', 'content', 'subject'])
test_sub.content_id = test.content_id
test_sub.content = test.content
test_sub.subject = test.subject
test_sub.to_csv('dataSet/test_subject.csv',index = False)

label1, label2 = list(), list()
for i in range(len(train)):
	label1.append(train.iloc[i]['subject'].split(','))
	label2.append(train.iloc[i]['sentiment_value'].split(','))

index = 0
train_senti = pd.DataFrame(columns = ['content_id', 'content', 'subject', 'sentiment_value'])
for i in range(len(train)):
	content_id = train.iloc[i]['content_id']
	content = train.iloc[i]['content']
	for sub, senti in zip(label1[i], label2[i]):
		train_senti = train_senti.append(pd.DataFrame({'content_id':content_id, 'content':content, 'subject':sub, 'sentiment_value':senti}, index = [index]))
		index += 1
train_senti.to_csv('dataSet/train_sentiment.csv',index = False)

label1, label2 = list(), list()
for i in range(len(test)):
	label1.append(test.iloc[i]['subject'].split(','))
	label2.append(test.iloc[i]['sentiment_value'].split(','))

index = 0
test_senti = pd.DataFrame(columns = ['content_id', 'content', 'subject', 'sentiment_value'])
for i in range(len(test)):
	content_id = test.iloc[i]['content_id']
	content = test.iloc[i]['content']
	for sub, senti in zip(label1[i], label2[i]):
		test_senti = test_senti.append(pd.DataFrame({'content_id':content_id, 'content':content, 'subject':sub, 'sentiment_value':senti}, index = [index]))
		index += 1
test_senti.to_csv('dataSet/test_local.csv', index = False)
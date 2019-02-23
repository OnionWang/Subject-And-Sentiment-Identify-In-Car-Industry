import tensorflow as tf
import sys
import modeling
import util
import numpy as np

from sklearn.model_selection import KFold


SEED = 2018

param = dict()
param['bert_config_path'] = 'chinese_L-12_H-768_A-12/bert_config.json'
param['checkpoint_path'] = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
param['train_subject_path'] = '../dataSet/train_subject.csv'
param['test_subject_path'] = '../dataSet/test_subject.csv'
param['output_path'] = '../result/subject_result.csv'
param['vocab_file'] = 'chinese_L-12_H-768_A-12/vocab.txt'
param['sentence_len'] = 80 # < 512


ids_train, masks_train, segment_ids_train, y_train, ids_test, masks_test, segment_ids_test, test_id, test_content, index2label = util.load_subject(param)
param['num_class'] = len(index2label)

# bert input
input_ids = tf.placeholder (shape = [None, param['sentence_len']], dtype = tf.int32, name = 'input_ids')
input_mask = tf.placeholder (shape = [None, param['sentence_len']], dtype = tf.int32, name = 'input_mask')
segment_ids = tf.placeholder (shape = [None, param['sentence_len']], dtype = tf.int32, name = 'segment_ids')
input_labels = tf.placeholder (shape = [None, param['num_class']], dtype = tf.float32, name = 'input_ids')
train_flag = tf.placeholder (dtype = tf.bool, name = 'is_training')
dropout_keep_prob = tf.placeholder(dtype = tf.float32, name = 'dropout_keep_prob')
learning_rate = tf.placeholder(dtype = tf.float32, name = 'learning_rate')

bert_config = modeling.BertConfig.from_json_file(param['bert_config_path'])
model = modeling.BertModel(
    config = bert_config,
    is_training = train_flag,
    input_ids = input_ids,
    input_mask = input_mask,
    token_type_ids = segment_ids,
    use_one_hot_embeddings = False # If you use TPU, set it True else False
)

output_layer = model.get_pooled_output() 
hidden_size = output_layer.shape[-1].value # 768

# your own full concect layer
output_weights = tf.get_variable('output_weights', [hidden_size, param['num_class']], initializer = tf.truncated_normal_initializer(stddev = 0.02))
output_bias = tf.get_variable('output_bias', [param['num_class']], initializer = tf.zeros_initializer())
with tf.variable_scope('loss'):
    output_layer = tf.nn.dropout(output_layer, keep_prob = dropout_keep_prob)
    logits = tf.matmul(output_layer, output_weights)
    logits = tf.nn.bias_add(logits, output_bias)

    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = input_labels, logits = logits)
    loss = tf.reduce_mean(losses)

    log_probs = tf.nn.sigmoid(logits)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list = [output_weights, output_bias])

# load bert checkpoint
tvars = tf.trainable_variables()
(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, param['checkpoint_path'])

tf.train.init_from_checkpoint(param['checkpoint_path'], assignment_map)

# tf.logging.info(''**** Trainable Variables ****')
# for var in tvars:
#     init_string = ''
#     if var.name in initialized_variable_names:
#         init_string = ', *INIT_FROM_CKPT*'
#     tf.logging.info('name = %s, shape = %s%s', var.name, var.shape, init_string)

fold = 5
epochs = 3
batch_size = 16

kfold = list(KFold(n_splits = fold, random_state = SEED, shuffle = True).split(ids_train, y_train))
y_target = np.zeros(y_train.shape)
y_test_pred = np.zeros((ids_test.shape[0], y_train.shape[1]))
for i, (train_index, val_index) in enumerate(kfold):
    input_idsList, input_masksList, segment_idsList = ids_train[train_index], masks_train[train_index], segment_ids_train[train_index]
    val_idsList, val_masksList, val_segment_idsList = ids_train[val_index], masks_train[val_index], segment_ids_train[val_index]
    y, val_y = y_train[train_index], y_train[val_index]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        text_num = len(input_idsList)
        iter_num = text_num  // batch_size
        iter_num = iter_num if text_num % batch_size == 0 else iter_num + 1
        lr_list = [0.01, 0.001, 0.0001]
        for epoch in range(epochs):
            print('='*80)
            lr = lr_list[epoch]
            if epoch == epochs - 1:
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            epoch_shuffle = np.random.permutation(np.arange(text_num))
            train_label = y[epoch_shuffle]
            train_idsList = input_idsList[epoch_shuffle]
            train_masksList = input_masksList[epoch_shuffle]
            train_segment = segment_idsList[epoch_shuffle]
            train_loss =  0.
            for i in range(iter_num):
                start = i * batch_size
                end = min((i + 1) * batch_size, text_num)
                batch_shuffle = np.random.permutation(np.arange(end - start))               
                batch_labels = train_label[start:end][batch_shuffle]
                batch_input_idsList = train_idsList[start:end][batch_shuffle]
                batch_input_masksList = train_masksList[start:end][batch_shuffle]
                batch_segment_idsList = train_segment[start:end][batch_shuffle]
                l, _ = sess.run([loss, train_op], feed_dict = {input_ids:batch_input_idsList, input_mask:batch_input_masksList, segment_ids:batch_segment_idsList, input_labels:batch_labels, learning_rate:lr, train_flag:True, dropout_keep_prob:0.9})
                train_loss += l
                if i % 50 == 0 and i > 0:
                    val_loss = sess.run(loss, feed_dict = {input_ids:val_idsList, input_mask:val_masksList, segment_ids:val_segment_idsList, input_labels:val_y, dropout_keep_prob:1.0})
                    print('iter_num = {}\ttrain loss = {:.5f}\tval_loss = {:.5f}'.format(i, train_loss/i, val_loss))
            y_pred = sess.run(log_probs, feed_dict = {input_ids:val_idsList, input_mask:val_masksList, segment_ids:val_segment_idsList, dropout_keep_prob:1.0})
            y_target[val_index] = y_pred
            test_pred = sess.run(log_probs, feed_dict = {input_ids:ids_test, input_mask:masks_test, segment_ids:segment_ids_test, dropout_keep_prob:1.0})
            y_test_pred += test_pred / fold
            threshold = util.search_threshold(np.squeeze(val_y), np.squeeze(y_pred))
            pre_label = util.get_label_pre(y_pred, threshold)
            val_label = util.get_label_real(val_y)
            f1 = util.compute_f1_subject(val_label, pre_label)
            print('Epoch\t%d/%d\tf1 score:%.6f:' %(epoch+1, epochs, f1))

y_target_pred = y_target.mean(axis = 1)
threshold = util.search_threshold(y_train, y_target_pred)
pre_label = util.get_label_pre(y_target_pred, threshold)
val_label = util.get_label_real(y_train)
f1 = util.compute_f1_subject(val_label, pre_label)
print('Final threshold:', threshold)
print('Final f1 score:%.6f' %f1)

test_logits = np.squeeze(y_test_pred)
test_label = util.get_label_pre(test_logits, threshold)
util.output_subject(test_id, test_content, test_label, index2label, param['output_path'])
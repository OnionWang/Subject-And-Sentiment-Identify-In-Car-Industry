import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Model
from keras.engine import Layer
from keras import initializers, layers, optimizers

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import  AdaBoostClassifier

SEED = 2018

def getModel(param, name):
    if name == 'FastText':
        model = FastText(param)
    elif name == 'TextCnn':
        model = TextCnn(param)
    elif name == 'TextRnn':
        model = TextRnn(param)
    elif name == 'TextRCnn':
        model = TextRCnn(param)
    elif name == 'LogisticRegression':
        model = OneVsRestClassifier(LogisticRegression(C = 0.5, random_state = SEED))
    elif name == 'SVM':
        model = OneVsRestClassifier(SVC(kernel = 'linear', probability = True, random_state = SEED))
    elif name == 'RandomForest':
        model = OneVsRestClassifier(RandomForestClassifier())
    elif name == 'GradientBoosting':
        model = OneVsRestClassifier(GradientBoostingClassifier())
    elif name == 'AdaBoost':
        model = OneVsRestClassifier(AdaBoostClassifier())
    elif name == 'AT_LSTM':
        model = AT_LSTM(param)
    elif name == 'GACE':
        model = GACE(param)
    elif name == 'HEAT':
        model = HEAT(param)

    return model


def FastText(param):
    inp = layers.Input(shape = (param['sentence_len'],))
    x = layers.Embedding(param['vocab_size'], param['embed_size'])(inp)
    x = layers.SpatialDropout1D(rate = 0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outp = layers.Dense(param['num_class'], activation = 'sigmoid')(x)
    model = Model(inputs = inp, outputs = outp)
    optimizer = optimizers.Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer)

    return model


def TextCnn(param):
    filter_sizes = [2, 3, 4]
    num_filters = 128

    inp = layers.Input(shape = (param['sentence_len'],))
    x = layers.Embedding(input_dim = param['vocab_size'], output_dim = param['embed_size'])(inp)
    x = layers.SpatialDropout1D(rate = 0.1)(x)
    maxpool_pool = []
    for filter_size in filter_sizes:
        conv = layers.Conv1D(num_filters, kernel_size = filter_size, activation = 'relu')(x)
        maxpool_pool.append(layers.GlobalMaxPooling1D()(conv))

    x = layers.Concatenate(axis = 1)(maxpool_pool)
    x = layers.Dropout(0.2)(x)
    outp = layers.Dense(units = param['num_class'], activation = 'sigmoid')(x)
    model = Model(inputs = inp, outputs = outp)
    optimizer = optimizers.Adam()
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer)
    
    return model


def TextRnn(param):
    hidden_size = 64
    inp = layers.Input(shape = (param['sentence_len'],))
    x = layers.Embedding(param['vocab_size'], param['embed_size'])(inp)
    x = layers.SpatialDropout1D(rate = 0.1)(x)
    x = layers.Bidirectional(layers.GRU(units = hidden_size, return_sequences = True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outp = layers.Dense(units = param['num_class'], activation = 'sigmoid')(x)
    model = Model(inputs = inp, outputs = outp)
    optimizer = optimizers.Adam()
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer)

    return model


def TextRCnn(param):
    hidden_size = 64
    inp = layers.Input(shape = (param['sentence_len'],))
    x = layers.Embedding(param['vocab_size'], param['embed_size'])(inp)
    x = layers.SpatialDropout1D(rate = 0.1)(x)
    x = layers.Bidirectional(layers.GRU(units = hidden_size, return_sequences = True))(x)
    x = layers.Conv1D(hidden_size, kernel_size = 2, activation = 'relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outp = layers.Dense(units = param['num_class'], activation = 'sigmoid')(x)
    model = Model(inputs = inp, outputs = outp)
    optimizer = optimizers.Adam()
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer)

    return model


# =================================================================================================================================== #
class Aspect_attention(Layer):
    def __init__(self, time_step, hidden_dim, aspect_dim, **kwargs):
        self.time_step = time_step
        self.hidden_dim = hidden_dim
        self.aspect_dim = aspect_dim
        super(Aspect_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[0][-1]
        aspect_dim = input_shape[1][-1]
        self.Wh = self.add_weight(name = 'Wh',
                                  initializer = 'glorot_uniform',
                                  shape = (hidden_dim, hidden_dim,))
        self.Wv = self.add_weight(name = 'Wv',
                                  initializer = 'glorot_uniform',
                                  shape = (aspect_dim, aspect_dim,))
        self.W = self.add_weight(name = 'W',
                                 initializer = 'glorot_uniform',
                                 shape = (hidden_dim + aspect_dim, 1))
        self.Wp = self.add_weight(name = 'Wp',
                                  initializer = 'glorot_uniform',
                                  shape = (hidden_dim, hidden_dim,))
        self.Wx = self.add_weight(name = 'Wx',
                                 initializer = 'glorot_uniform',
                                 shape = (hidden_dim, hidden_dim,))
        super(Aspect_attention, self).build(input_shape)

    def call(self, x):
        hidden_dim = self.hidden_dim
        aspect_dim = self.aspect_dim
        time_step = self.time_step
        # [batch_size, sentence_len, lstm_hidden_size] ------> [batch_size * sentence_len, lstm_hidden_size]
        hidden = K.reshape(x[0], (-1, hidden_dim))
        # [batch_size, sentence_len, embed_size] ------> [batch_size * sentence_len, embed_size]
        aspect = K.reshape(x[1], (-1, aspect_dim))
        # [batch_size * sentence_len, lstm_hidden_size + embed_size]
        M = K.tanh(K.concatenate((K.dot(hidden, self.Wh), K.dot(aspect, self.Wv)), axis = 1))
        # [batch_size, sentence_len]
        alpha = K.softmax(K.reshape(K.dot(M, self.W), (-1, time_step, 1)), axis = 1)
        # [batch_size, lstm_hidden_size]
        r = K.sum(x[0] * alpha, axis = 1)
        # [batch_size, lstm_hidden_size]
        h = K.tanh(K.dot(r, self.Wp) + K.dot(K.reshape(x[0][:, -1, :], (-1, hidden_dim)), self.Wx))

        return h

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


def AT_LSTM(param):
    hidden_size = 128
    inp1 = layers.Input(shape = (param['sentence_len'],))
    x1 = layers.Embedding(input_dim = param['vocab_size'], output_dim = param['embed_size'])(inp1)
    x1 = layers.SpatialDropout1D(rate = 0.2)(x1)
    x1 = layers.Bidirectional(layers.LSTM(units = hidden_size, return_sequences = True))(x1)

    inp2 = layers.Input(shape = (1,))
    x2 = layers.Embedding(input_dim = param['num_subject'], output_dim = param['embed_size'])(inp2)
    x2 = layers.Flatten()(x2)
    x2 = layers.RepeatVector(param['sentence_len'])(x2)

    x = Aspect_attention(param['sentence_len'], hidden_size * 2, param['embed_size'])([x1, x2])
    x = layers.Dropout(0.1)(x)
    outp = layers.Dense(units = param['num_class'], activation = 'softmax')(x)
    model = Model(inputs = [inp1, inp2], outputs = outp)
    optimizer = optimizers.Adam()
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
    
    return model

def myconcat(x):
    return K.concatenate(x, axis = 2)

def mycompute_shape(input_shape):
    shape = list(input_shape[0])
    shape[-1] += input_shape[1][-1]
    return tuple(shape)

def AE_LSTM(param):
    hidden_size = 64
    inp1 = layers.Input(shape = (param['sentence_len'],))
    x1 = layers.Embedding(input_dim = param['vocab_size'], output_dim = param['embed_size'])(inp1)
      
    inp2 = layers.Input(shape = (1,))
    x2 = layers.Embedding(input_dim = param['num_subject'], output_dim = param['embed_size'])(inp2)
    x2 = layers.Flatten()(x2)
    x2 = layers.RepeatVector(param['sentence_len'])(x2)

    x = layers.Lambda(myconcat, output_shape = mycompute_shape)([x1, x2])
    # x = layers.SpatialDropout1D(rate = 0.1)(x)
    x = layers.Bidirectional(layers.LSTM(units = hidden_size, return_sequences = False))(x)

    x = layers.Dropout(0.1)(x)
    outp = layers.Dense(units = param['num_class'], activation = 'softmax')(x)
    model = Model(inputs = [inp1, inp2], outputs = outp)
    optimizer = optimizers.Adam()
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
    
    return model


class Aspect_conv(Layer):
    def __init__(self, **kwargs):
        super(Aspect_conv, self).__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[1][-1]
        aspect_dim = input_shape[2][-1]
        self.W = self.add_weight(name = 'W',
                                  initializer = 'glorot_uniform',
                                  shape = (aspect_dim, hidden_dim,))
        super(Aspect_conv, self).build(input_shape)

    def call(self, x):
        conv1, conv2, aspect = x

        a = K.relu(conv2 + K.dot(aspect, self.W))
        out = conv1 * a

        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def GCAE(param):
    filter_sizes = [2, 3, 4]
    num_filters = 128

    inp1 = layers.Input(shape = (param['sentence_len'],))
    x1 = layers.Embedding(input_dim = param['vocab_size'], output_dim = param['embed_size'])(inp1)
    # x1 = layers.SpatialDropout1D(rate = 0.2)(x1)

    inp2 = layers.Input(shape = (1,))
    x2 = layers.Embedding(input_dim = param['num_subject'], output_dim = param['embed_size'])(inp2)
    x_fla = layers.Flatten()(x2)
    maxpool_pool = []
    for filter_size in filter_sizes:
        conv1 = layers.Conv1D(num_filters, kernel_size = filter_size, activation = 'tanh')(x1)
        conv2 = layers.Conv1D(num_filters, kernel_size = filter_size, activation = None)(x1)
        x2 = layers.RepeatVector(param['sentence_len'] - filter_size + 1)(x_fla)
        conv = Aspect_conv()([conv1, conv2, x2])
        maxpool_pool.append(layers.GlobalMaxPooling1D()(conv))

    z = layers.Concatenate(axis = 1)(maxpool_pool)   
    z = layers.Dropout(0.1)(z)
    outp = layers.Dense(param['num_class'], activation = 'softmax')(z)
    model = Model(inputs = [inp1, inp2], outputs = outp)
    optimizer = optimizers.Adam()
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
    
    return model

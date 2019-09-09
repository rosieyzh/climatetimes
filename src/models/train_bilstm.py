import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, Bidirectional, GlobalMaxPool1D, TimeDistributed, Flatten
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.keras.callbacks import History 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

from utils import preprocess

#globals
EMBEDDING_DIM=300
MAX_SEQUENCE_LEN = 1000
MAX_NUM_WORDS = 100000
RATE_DROP_LSTM = 0.15
RATE_DROP_DENSE= 0.1
BASE_DIR = './saved_models'
GLOVE_FILE = os.path.join(BASE_DIR, 'glove.6B/glove.6B.300d.txt')
GOOGLE_FILE = os.path.join(BASE_DIR, 'GoogleNews-vectors-negative300.bin')
OUR_FILE = os.path.join(BASE_DIR, 'final_w2v/w2v_1_both')

FILEPATH_LEFT = '../../data/all_left_filtered.csv'
FILEPATH_RIGHT = '../../data/all_right_filtered.csv'

np.random.seed(2019) #for reproducibility
#Attention Layer
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim

def get_coefs(word,*arr): 
        #For creating embedding matrix for different models
        return word, np.asarray(arr, dtype='float32')

def train_bilstm(features, labels, embedding_type):
        labels_index =  {0:0,1:1}
        labels = to_categorical(np.asarray(labels))
        #split data into 80% train, 10% validation, and 10% test
        xTrain, xVal, yTrain, yVal = train_test_split(features,labels, test_size = 0.2, random_state = 0)
        #xVal, xTest, yVal, yTest = train_test_split(xVal,yVal, test_size = 0.5, random_state = 0)

        embedding_file = ''
        if embedding_type == 'glove':
                embedding_file = GLOVE_FILE
        elif embedding_type == 'google':
                embedding_file = GOOGLE_FILE
        elif embedding_type == 'ours':
                embedding_file = OUR_FILE

        #Tokenize sentences
        tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
        tokenizer.fit_on_texts(xTrain)
        xTrain = tokenizer.texts_to_sequences(xTrain)
        xVal = tokenizer.texts_to_sequences(xVal)
        #xTest = tokenizer.texts_to_sequences(xTest)

        #Pad articles with length less than MAX_SEQUENCE_LEN
        xTrain = pad_sequences(xTrain, maxlen = MAX_SEQUENCE_LEN)
        xVal = pad_sequences(xVal, maxlen = MAX_SEQUENCE_LEN)
        #xTest = pad_sequences(xTest, maxlen = MAX_SEQUENCE_LEN)

        #Create embedding index
        word_index = tokenizer.word_index
        num_words = min( MAX_NUM_WORDS, len(word_index)+1) 
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        

        if type=='glove':
                embeddings_index = index_glove()
                print("Preparing embedding matrix")
                
                for word, i in word_index.items():
                    if i >=  MAX_NUM_WORDS:
                        continue
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        #If vector returns none, zeroes are placed instead
                        embedding_matrix[i] = embedding_vector

        elif type=='google':
                
                print("Using Google pretrained word embeddings.")
                word_vectors = KeyedVectors.load_word2vec_format('./saved_models/GoogleNews-vectors-negative300.bin', binary=True)

                for word, i in word_index.items():
                    if i >=  MAX_NUM_WORDS:
                        continue
                    if word in word_vectors.wv.vocab:
                        embedding_vector = word_vectors[word]
                    if embedding_vector is not None:
                        #If vector returns none, zeroes are placed instead
                        embedding_matrix[i] = embedding_vector
                        

        elif type=='ours':
                
                print("Using our trained word embeddings.")
                word_vectors = KeyedVectors.load('./saved_models/final_w2v/w2v_1_both')

                for word, i in word_index.items():
                    if i >=  MAX_NUM_WORDS:
                        continue
                    if word in word_vectors.wv.vocab:
                        embedding_vector = word_vectors[word]
                    if embedding_vector is not None:
                        #If vector returns none, zeroes are placed instead
                        embedding_matrix[i] = embedding_vector

        #Model architecture
        embedding_layer = Embedding(num_words,
                                            EMBEDDING_DIM,
                                            weights=[embedding_matrix],
                                            input_length=MAX_SEQUENCE_LEN,
                                            trainable=False)

        sequence_input=Input(shape=(MAX_SEQUENCE_LEN,), dtype='int32')

        embedded_sequences=embedding_layer(sequence_input)
        x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(embedded_sequences)
        #x = Attention(step_dim=MAX_SEQUENCE_LEN)(x)
        x = Flatten() (x)
        x = Dense(64,activation='relu')(x)
        x = Dropout(RATE_DROP_DENSE)(x)
        x = Dense(len(labels_index), activation="sigmoid")(x)
        model = Model(inputs=sequence_input, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['acc'])

        #Implement Early Stopping
        history = History()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(xTrain, yTrain,
                  batch_size=16,
                  epochs=24,
                  validation_data=(xVal, yVal), callbacks=[early_stopping, history])

        pred_val_y = model.predict([xVal], batch_size=16, verbose=1)
        for thresh in np.arange(0.1, 0.501, 0.01):
            thresh = np.round(thresh, 2)
            print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(yVal, (pred_val_y>thresh).astype(int), average='micro')))
        #pred_test_y = model.predict([xTest], batch_size=32, verbose=1)
        
        #model.save('./saved_models/classifiers/bilstm_earlystop_{}_{}'.format(type, pred_test_y))

        #plotting
        plt.figure(1)  

        # summarize history for accuracy  
        plt.subplot(211)  
        plt.plot(history.history['acc'])  
        plt.plot(history.history['val_acc'])  
        plt.title('model accuracy')  
        plt.ylabel('accuracy')    
        plt.legend(['train', 'test'], loc='upper left')  
           
        # summarize history for loss  
        plt.subplot(212)  
        plt.plot(history.history['loss'])  
        plt.plot(history.history['val_loss'])  
        plt.title('model loss')  
        plt.ylabel('loss')  
        plt.xlabel('epoch')  
        plt.legend(['train', 'test'], loc='upper left')  
        plt.savefig('bilstm_earlystop_{}.png'.format(type, bbox_inches='tight')) 

if __name__ == '__main__':
        left_data = pd.read_csv(FILEPATH_LEFT)
        right_data = pd.read_csv(FILEPATH_RIGHT)

        left_articles=left_data['content'].tolist()
        right_articles=right_data['content'].tolist()
        all_articles=left_articles+right_articles
        all_labels = left_data['denial?'].tolist() + right_data['denial?'].tolist()

        preprocessed_articles=[' '.join(article) for article in preprocess(all_articles)]

        train_bilstm(preprocessed_articles, all_labels, 'ours')

        train_bilstm(preprocessed_articles, all_labels, 'google')

        train_bilstm(preprocessed_articles, all_labels, 'glove')




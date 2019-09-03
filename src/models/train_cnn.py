from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import History 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
from tensorflow.keras import optimizers

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors

from utils import preprocess

import matplotlib.pyplot as plt 

#global variables
BASE_DIR = './saved_models'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
LEARNING_RATE = .00011
BATCH_SIZE = 32
DROPOUT_RATE = 0.45
INNERLAYER_DROPOUT_RATE = 0.15

def index_glove():

	print("Indexing word vectors:")
	embeddings_index={}
	with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt')) as f:
		for line in f:
			values = line.split()
			word =values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	print("Found %s word vectors" % len(embeddings_index))
	return embeddings_index




def train_cnn(features, labels, type, max_sequence_len = MAX_SEQUENCE_LENGTH, max_nb_words=MAX_NUM_WORDS, embedding_dim=EMBEDDING_DIM, validation_split=VALIDATION_SPLIT, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, droupout_rate=DROPOUT_RATE, innerlayer_dropout_rate=INNERLAYER_DROPOUT_RATE):
		'''
			Trains 1-d CNN on climate change news articles.
			Referenced from https://www.microsoft.com/developerblog/2017/12/04/predicting-stock-performance-deep-learning/
		'''
		
		labels_index =  {0:0,1:1}

		#tokenize text
		tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
		tokenizer.fit_on_texts(features)
		sequences = tokenizer.texts_to_sequences(features)

		word_index = tokenizer.word_index
		print('Found %s unique tokens.' % len(word_index))

		data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

		labels = to_categorical(np.asarray(labels))
		print('Shape of data tensor:', data.shape)
		print('Shape of label tensor:', labels.shape)

		# split the data into a training set and a validation set
		indices = np.arange(data.shape[0])
		np.random.shuffle(indices)
		data = data[indices]
		labels = labels[indices]
		num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

		x_train = data[:-num_validation_samples]
		y_train = labels[:-num_validation_samples]
		x_val = data[-num_validation_samples:]
		y_val = labels[-num_validation_samples:]

		num_words = min(max_nb_words, len(word_index)+1) 
		embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
		

		if type=='glove':
			embeddings_index = index_glove()
			print("Preparing embedding matrix")
			

			for word, i in word_index.items():
			    if i >= MAX_NUM_WORDS:
			        continue
			    embedding_vector = embeddings_index.get(word)
			    if embedding_vector is not None:
			    	#If vector returns none, zeroes are placed instead
			        embedding_matrix[i] = embedding_vector

		elif type=='google':

			print("Using Google pretrained word embeddings.")
			word_vectors = KeyedVectors.load_word2vec_format('./saved_models/GoogleNews-vectors-negative300.bin', binary=True)

			for word, i in word_index.items():
			    if i >= MAX_NUM_WORDS:
			        continue
			    
			    embedding_vector = word_vectors[word]
			    if embedding_vector is not None:
			    	#If vector returns none, zeroes are placed instead
			    	embedding_matrix[i] = embedding_vector
			    	

		elif type=='ours':

			print("Using our trained word embeddings.")
			word_vectors = KeyedVectors.load('./saved_models/final_w2v/w2v_1_both')

			for word, i in word_index.items():
			    if i >= MAX_NUM_WORDS:
			        continue
			    try:
			    	embedding_vector = word_vectors[word]
			    if embedding_vector is not None:
			    	#If vector returns none, zeroes are placed instead
			    	embedding_matrix[i] = embedding_vector


		# load pre-trained word embeddings into an Embedding layer
		# note that we set trainable = False so as to keep the embeddings fixed
		embedding_layer = Embedding(num_words,
		                            EMBEDDING_DIM,
		                            embeddings_initializer=Constant(embedding_matrix),
		                            input_length=MAX_SEQUENCE_LENGTH,
		                            trainable=False)

		sequence_input=Input(shape=(max_sequence_len,), dtype='int32')

		embedded_sequences=embedding_layer(sequence_input)

		#Model architecture
		x = Conv1D(128, 5, activation='elu', kernel_initializer='lecun_uniform')(embedded_sequences)
		x = MaxPooling1D(5)(x)
		x = Dropout(innerlayer_dropout_rate)(x)
		 
		x = Conv1D(128, 5, activation='elu', kernel_initializer='lecun_uniform')(x)
		x = MaxPooling1D(5)(x)
		x = Dropout(innerlayer_dropout_rate)(x)
		 
		x = Conv1D(128, 5, activation='elu', kernel_initializer='lecun_uniform')(x)
		x = MaxPooling1D(35)(x)  # global max pooling
		 
		x = Flatten()(x)
		x = Dense(100, activation='elu', kernel_initializer='lecun_uniform')(x) # best initializers: #glorot_normal #VarianceScaling #lecun_uniform
		x = Dropout(innerlayer_dropout_rate)(x)
		 
		preds = Dense(len(labels_index), activation='softmax')(x) #no initialization in output layer
		 
		model = Model(sequence_input, preds)

		adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=0.5)#, clipnorm=1.)
		rmsprop = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-08, decay=0.00)

		model.compile(loss='categorical_crossentropy',
		              optimizer= adam,
		              metrics=['acc'])

		history = History()

		early_stopping = EarlyStopping(monitor='val_loss', patience=10)

		history = model.fit(x_train, y_train,
		          batch_size=32,
		          epochs=24,
		          validation_data=(x_val, y_val), callbacks=[early_stopping, history])


		model.save('./saved_models/classifiers/cnn_earlystop_{}'.format(type))


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
		plt.savefig('cnn_earlystop_{}.png'.format(type, bbox_inches='tight'))  

if __name__ == '__main__':

	FILEPATH_LEFT = '../../data/all_left_filtered.csv'
	FILEPATH_RIGHT = '../../data/all_right_filtered.csv'
	FILEPATH_GOOGLE_PRETRAINED='./saved_models/GoogleNews-vectors-negative300.bin'
	FILEPATH_BOTH = './saved_models/final_w2v/w2v_1_both'


	left_data = pd.read_csv(FILEPATH_LEFT)
	right_data = pd.read_csv(FILEPATH_RIGHT)

	left_articles=left_data['content'].tolist()
	right_articles=right_data['content'].tolist()
	all_articles=left_articles+right_articles
	all_labels = left_data['denial?'].tolist() + right_data['denial?'].tolist()

	preprocessed_articles=preprocess(all_articles)

	train_cnn(preprocessed_articles, all_labels, 'glove')

	train_cnn(preprocessed_articles, all_labels, 'google')
	
	train_cnn(preprocessed_articles, all_labels, 'ours')










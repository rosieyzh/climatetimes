import numpy as np
import pandas as pd
import nltk
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.phrases import Phraser, Phrases

import simplejson
from utils import preprocess, train_rf, train_svm

def get_w2v_individual(article, model):
	'''
		Retrieves pretrained vector for article title based on pretrained Word2Vec Google News model.
		Input: title - tokenized title for individual article
		Output: vector - w2v vector representation of title
	'''
	#load Google News
	vector = np.mean(np.array([model[word] for word in article if word in model.vocab]), axis=0)
	return vector



def get_w2v_all(articles, labels, pretrained_file_path, n=100):
	'''
		Retrieves pretrained vector for all articles in corpus based on pretrained Word2Vec Google News model.
		Input: articles - tokenized titles for all articles
		Output: w2v_articles - w2v vector representation of each article
	'''

	preprocessed_articles=preprocess(articles)
	shortened_articles = [article[:n-1] if len(article) >=n else article for article in preprocessed_articles]

	model = KeyedVectors.load_word2vec_format(pretrained_file_path, binary=True)
	w2v_articles = np.zeros((14252,300))
	indices_rm =[]
	num_rm=0
	for i in range(len(shortened_articles)):
		vector = get_w2v_individual(shortened_articles[i], model)
		if np.all(np.isnan(vector)):
			indices_rm.append(i)
			num_rm+=1
		else:
			w2v_articles[i-num_rm]=vector
	print(indices_rm)
	good_labels = rm_labels(labels, indices_rm)
	return w2v_articles[:len(good_labels)], np.array(good_labels)

def rm_labels(labels, indices):
	'''
		Removes indices of labels for titles that didn't return a word embedding vector.
	'''
	shift=0
	for index in indices:
		del labels[index-shift]
		shift+=1
	return labels

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

	# CLASSIFICATION WITH FIRST 100, 500, AND 1000 WORDS using pretrained Google News
	n_values = [100, 500, 1000]
	
	print("Using Google pretrained")
	w2v_articles, w2v_labels = get_w2v_all(all_articles, all_labels, FILEPATH_GOOGLE_PRETRAINED, 1000)
	print(w2v_articles.shape)
	print(w2v_labels.shape)

	train_rf(w2v_articles, w2v_labels, 'w2v_google_1000')
	print("Finished rf training")
	
	train_svm(w2v_articles, w2v_labels,  'w2v_google_1000')
	print("Finished svm training")
	
	'''
	for n in n_values:
		print("Using our trained model")
		w2v_articles, w2v_labels = get_w2v_all(all_articles, all_labels, FILEPATH_BOTH, n)
		print(w2v_articles.shape)
		print(w2v_labels.shape)

		train_rf(w2v_articles, w2v_labels, 'w2v_ours_{}'.format(n))
		print("Finished rf training")

		train_svm(w2v_articles, w2v_labels,  'w2v_ours_{}'.format(n))
		print("Finished svm training")
	'''

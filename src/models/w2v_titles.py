import numpy as np
import pandas as pd
import nltk
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.phrases import Phraser, Phrases

from utils import preprocess, train_rf, train_svm


'''
	Retrieves feature representation of news articles using Word2Vec models pretrained on Google News
	as well as newly trained Word2Vec model.
'''

def train_w2v(articles, side):
	'''
		Trains word2vec model from scratch based on articles.

		Input:
			articles - list of article content read from csv
			side - string indicating 'left' or 'right'
	'''

	preprocessed_articles = preprocess(articles, ngrams=True)

	#train word2vec model
	w2v = gensim.models.word2vec.Word2Vec(sentences=preprocessed_articles, 
										  min_count=3,
										  size=300,
										  workers=32,
										  window=5
										  )
	print("Word2Vec Model finished training")

	if 'global_warming' in w2v.wv.vocab:
		print(w2v.wv.most_similar('global_warming'))
	else:
		print('global_warming not in vocab')

	#save model and word vectors
	w2v.save('./saved_models/w2v_{}'.format(side))
	w2v.wv.save_word2vec_format("./saved_models/w2v_vectors_{}".format(side))
	print("Model saved successfully!")


def get_w2v_individual(title, model):
	'''
		Retrieves pretrained vector for article title based on pretrained Word2Vec Google News model.
		Input: title - tokenized title for individual article
		Output: vector - w2v vector representation of title
	'''
	#load Google News
	vector = np.mean(np.array([model[word] for word in title if word in model.vocab]), axis=0)
	return vector



def get_w2v_all(titles, pretrained_file_path):
	'''
		Retrieves pretrained vector for all titles in corpus based on pretrained Word2Vec Google News model.
		Input: titles - tokenized titles for all articles
		Output: w2v_titles - w2v vector representation of each title
	'''

	preprocessed_titles=preprocess(titles)
	model = KeyedVectors.load_word2vec_format(pretrained_file_path, binary=True)
	w2v_titles = np.zeros((14247,300))
	indices_rm =[]
	num_rm=0
	for i in range(len(preprocessed_titles)):
		vector = get_w2v_individual(preprocessed_titles[i], model)
		if np.all(np.isnan(vector)):
			indices_rm.append(i)
			num_rm+=1
		else:
			w2v_titles[i-num_rm]=vector
	print(w2v_titles.shape)
	return w2v_titles, indices_rm

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

	left_data = pd.read_csv(FILEPATH_LEFT)
	right_data = pd.read_csv(FILEPATH_RIGHT)

	left_articles=left_data['content'].tolist()
	right_articles=right_data['content'].tolist()

	'''
	#TRAIN W2V MODELS
	train_w2v(left_articles, 'left')
	train_w2v(right_articles, 'right')
	print("Finished w2v training")
	'''

	#GET PRETRAINED W2V VECTORS
	all_titles = left_data['title'].tolist() + right_data['title'].tolist()

	w2v_titles, indices_rm = get_w2v_all(all_titles, FILEPATH_GOOGLE_PRETRAINED)
	print("Retrieved pretrained embeddings for all titles")
	print(indices_rm)

	#TRAIN MODELS
	all_labels = left_data['denial?'].tolist() + right_data['denial?'].tolist()
	all_labels = np.array(rm_labels(all_labels, indices_rm))
	print(all_labels.shape)

	train_rf(w2v_titles, all_labels, 'w2v')
	print("Finished rf training")
	train_svm(w2v_titles, all_labels, 'w2v')
	print("Finished svm training")




	



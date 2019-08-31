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

	print(w2v.wv.most_similar('global_warming'))

	#save model and word vectors
	w2v.save('./saved_models/w2v_{}'.format(side))
	w2v.wv.save_word2vec_format("./saved_models/w2v_vectors_{}".format(side))
	print("Model saved successfully!")


def get_w2v_individual(title, pretrained_file_path):
	'''
		Retrieves pretrained vector for article title based on pretrained Word2Vec Google News model.
		Input: title - tokenized title for individual article
		Output: vector - w2v vector representation of title
	'''
	#load Google News
	model = KeyedVectors.load_word2vec_format(pretrained_file_path, binary=True)
	vector = [model[word] for word in title]

	return vector



def get_w2v_all(titles):
	'''
		Retrieves pretrained vector for all titles in corpus based on pretrained Word2Vec Google News model.
		Input: titles - tokenized titles for all articles
		Output: w2v_titles - w2v vector representation of each title
	'''

	preprocessed_titles=preprocess(titles)
	w2v_titles=[]
	for title in preprocessed_titles:
		w2v_titles.append(get_w2v_individual(title, '../../data/GoogleNews-vectors-negative300.bin'))

	return w2v_titles



if __name__ == '__main__':
	FILEPATH_LEFT = '../../data/all_left_filtered.csv'
	FILEPATH_RIGHT = '../../data/all_right_filtered.csv'

	left_data = pd.read_csv(FILEPATH_LEFT)
	right_data = pd.read_csv(FILEPATH_RIGHT)

	left_articles=left_data['content'].tolist()
	right_articles=right_data['content'].tolist()


	'''
	#TRAIN W2V MODELS
	train_w2v(left_articles, 'left')
	train_w2v(right_articles, 'right')
	'''

	#GET PRETRAINED W2V VECTORS
	all_titles = left_data['title'].tolist() + right_data['title'].tolist()
	w2v_titles = get_w2v_all(all_titles)

	#TRAIN MODELS
	all_labels = left_data['denial?'].tolist() + right_data['denial?'].tolist()

	train_rf(w2v_titles, all_labels)
	train_svm(w2v_titles, all_labels)


	



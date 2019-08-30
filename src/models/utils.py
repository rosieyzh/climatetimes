import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
import gensim
from gensim.models.phrases import Phraser, Phrases
import re

def preprocess(data, ngrams=False):
	'''
		Input:
			data - List of articles/titles
			ngrams - Train own n-grams using Gensim's Phraser or incorporate pretrained (applies to Google News Word2Vec)
		Output:
			List of tokenized words for each title/article
	'''

	#remove links
	processed_data = [re.sub(r'^https?:\/\/.*?[\r\n\s]+', '', article, flags=re.MULTILINE) for article in data]

	#remove punctuation
	#tokenize by word
	tokenizer = RegexpTokenizer(r'\w+')
	processed_data = [tokenizer.tokenize(article) for article in processed_data]

	#remove stopwords
	stop_words = stopwords.words('english')
	rm_stop = [[word for word in article if word.lower() not in stop_words] for article in processed_data]

	#incorporate bigrams and trigrams
	if ngrams:
		bigram = Phrases(rm_stop, min_count=5, threshold=100)
		trigram = Phrases(bigram[rm_stop], threshold=100)

		bigram_mod = Phraser(bigram)
		trigram_mod = Phraser(trigram)

		#with_bigram = [bigram_mod[article] for article in processed_data]
		with_trigram = [trigram_mod[bigram_mod[article]] for article in rm_stop]

		return with_trigram

	return rm_stop
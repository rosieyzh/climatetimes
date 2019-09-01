import numpy as np
import pandas as pd

#for preprocessing
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
import gensim
from gensim.models.phrases import Phraser, Phrases
import re

#for baseline models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

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

		with_bigram = [bigram_mod[article] for article in rm_stop]
		with_trigram = [trigram_mod[article] for article in with_bigram]

		return with_trigram

	return rm_stop



def train_rf(features, labels, type):
	'''
		Implements random forest with Grid Search for hyperparameter optimization.

		Input:
			features - preprocessed articles outputted by preprocess()
			labels - 0 or 1 assigned to each article
		Output:
			best_model: saved pickle model file
	'''
	xTrain, xTest, yTrain, yTest = train_test_split(features,labels, test_size = 0.2, random_state = 0)
	forest = RandomForestClassifier(random_state = 0)

	#hyperparameters to test with GridSearch
	n_estimators = [100, 300, 500, 800, 1200]
	max_depth = [5, 8, 15, 25, 30]
	min_samples_split = [2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10] 

	#Create dictionary for GridSearch to iterate over
	parameters = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

	#Instantiate grid search
	forest_grid = GridSearchCV(estimator = forest, param_grid = parameters, cv = 3, verbose = 1, n_jobs = -1)

	#Fit grid search to data
	forest_grid.fit(xTrain, yTrain)

	#Print the parameters of the best model
	print(forest_grid.best_params_)
	best_forest = forest_grid.best_estimator_

	#Test best model
	predicted = best_forest.predict(xTest)
	accuracy = np.mean(predicted == yTest)*100
	print("Model accuracy on test set: {}".format(accuracy))

	#Save model
	with open('./saved_models/best_forest_{}.pb'.format(type), 'wb') as f:
		pickle.dump(best_forest, f)



def train_svm(features,labels, type):
	'''
		Implements SVM with Grid Search for hyperparameter optimization.

		Input:
			features - preprocessed articles outputted by preprocess()
			labels - 0 or 1 assigned to each article
			type - 'w2v', 'lda', 'd2v', 'wmd'
		Output:
			best_model: saved pickle model file
	'''
	xTrain, xTest, yTrain, yTest = train_test_split(features,labels, test_size = 0.2, random_state = 0)
	svm = SVC(random_state = 0)

	#Create dictionary for GridSearch to iterate over
	parameters = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf', 'poly']}

    #Instantiate grid search
	svm_grid = GridSearchCV(estimator = svm, param_grid = parameters, cv = 3, verbose = 1, n_jobs = -1)

    #Fit grid search to data 
	svm_grid.fit(xTrain, yTrain)
	
    #Print the parameters of the best model
	print(svm_grid.best_params_)
	best_svm = svm_grid.best_estimator_

	#Test best model
	predicted = best_svm.predict(xTest)
	accuracy = np.mean(predicted == yTest)*100
	print("Model accuracy on test set: {}".format(accuracy))

	#Save model
	with open('./saved_models/best_svm_{}.pb'.format(type), 'wb') as f:
		pickle.dump(best_svm, f)

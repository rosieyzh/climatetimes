import numpy as np
import pandas as pd
import nltk
import gensim
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases


'''
	Retrieves feature representation of news articles using Word2Vec models pretrained on Google News
	as well as newly trained Word2Vec model.
'''
FILEPATH_LEFT = '../../data/all_left.csv'
FILEPATH_RIGHT = '../../data/all_right.csv'

left_data = pd.read_csv(FILEPATH_LEFT)
right_data = pd.read_csv(FILEPATH_RIGHT)




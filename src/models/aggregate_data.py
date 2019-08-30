import pandas as pd

def agg_data(files, saveFile):
	li=[]
	for filename in files:
		df = pd.read_csv(filename, index_col=None, header=0)
		li.append(df)
	
	all_data = pd.concat(li, axis=0, ignore_index=True)
	all_data.to_csv('../../data/{}.csv'.format(saveFile))

if __name__ == '__main__':
	right = [
			'../../data/right/data_climate_depot.csv',
			'../../data/right/data_fox_articles.csv',
			'../../data/right/data_global_climate_scam.csv',
			'../../data/right/data_breitbart.csv',
			'../../data/right/data_wattsup.csv'
			]
	left = [
			'../../data/left/data_abc_articles.csv',
			'../../data/left/data_alternet_articles.csv',
			'../../data/left/data_cbs_articles.csv',
			'../../data/left/data_democracynow_articles.csv',
			'../../data/left/data_mj_articles.csv',
			'../../data/left/data_inside_climate_news_articles.csv'
		   ]

	agg_data(right, 'all_right')
	agg_data(left, 'all_left')



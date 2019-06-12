from Models import Shared
from sklearn.manifold import TSNE
import sys
import glob
import natsort
import pandas as pd
import numpy as np
import random

## Usage: python tsne_s1 <dataset_dir> <perplexity>
## Ex: python tsne_s1 ./Datasets/gaussians 30
if __name__ == '__main__':
	dataset_dir = sys.argv[1]
	perplexity = float(sys.argv[2])
	# print(dataset_dir, perplexity)

	Xs = []
	y = []
	n_revisions = 0
	if 'quickdraw' in dataset_dir or 'fashion' in dataset_dir or 'faces' in dataset_dir:
		X, info_df, n_revisions, CATEGORIES = Shared.load_drawings(dataset_dir + '/')
		N = len(X)
		X_flat = np.reshape(np.ravel(X), (N, -1))
		for t, df in info_df.groupby('t'):
			df = df.sort_values(['drawing_cat_id', 'drawing_id'])
			if len(y) == 0:
				y = df['drawing_cat_str'].str.cat(df['drawing_id'].astype(str), sep='-')
			Xs.append(X_flat[df.X_index])
	else:
		csvs = natsort.natsorted(glob.glob(dataset_dir + '/*'))
		n_revisions = len(csvs)
		for csv in csvs:
			df = pd.read_csv(csv, index_col=0)
			if len(y) == 0:
				y = df.index
			Xs.append(df.values)

	# for a in y:
	# 	print(a)
	# 	sys.stdout.flush()

	try:
		columns = ['id']
		df_out = pd.DataFrame(index=y)
		for t, X in enumerate(Xs):
			print(t, end=' ')
			sys.stdout.flush()
			random_state = random.randint(0, 100)
			# Reduced number of epochs and sigma_iters. Previous were 1000 and 50.
			# Y = tsne(X, perplexity=perplexity, n_epochs=200, sigma_iters=20, random_state=random_state, verbose=0)
			Y = TSNE(n_components=2, perplexity=perplexity, verbose=0, random_state=random_state).fit_transform(X)
			df_out['t{}d0'.format(t)] = Y[:,0]
			df_out['t{}d1'.format(t)] = Y[:,1]
		if (len(df_out) - df_out.count()).sum() == 0:
			df_out.to_csv('./Output/{}-tsne_s1_{}p.csv'.format(dataset_dir.split('/')[-1], int(perplexity)), index_label='id')
			print(' '.join(sys.argv), 'OK')
		else:
			print(' '.join(sys.argv), 'crashed')
	except:
		print(' '.join(sys.argv), 'crashed')

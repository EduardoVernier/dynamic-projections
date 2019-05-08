from tsne import tsne
import sys
import glob
import natsort
import pandas as pd
import numpy as np
from Models import Shared

## Usage: python tsne_s1 <dataset_dir> <perplexity>
## Ex: python tsne_s1 ../../Datasets/gaussians 30
if __name__ == '__main__':
	dataset_dir = sys.argv[1]
	perplexity = float(sys.argv[2])
	# print(dataset_dir, perplexity)

	X = []
	y = []
	n_revisions = 0
	n_points = 0
	if 'quickdraw' in dataset_dir or 'fashion' in dataset_dir:
		X_drawing, info_df, n_revisions, CATEGORIES = Shared.load_drawings(dataset_dir + '/')
		N = len(X_drawing)
		X_flat = np.reshape(np.ravel(X_drawing), (N, -1))
		for t, df in info_df.groupby('t'):
			df = df.sort_values(['drawing_cat_id', 'drawing_id'])
			if len(y) == 0:
				y = df['drawing_cat_str'].str.cat(df['drawing_id'].astype(str), sep='-')
				n_points = len(y)
			for p in X_flat[df.X_index]:
				X.append(p)
	else:
		csvs = natsort.natsorted(glob.glob(dataset_dir + '/*'))
		n_revisions = len(csvs)
		for csv in csvs:
			df = pd.read_csv(csv, index_col=0)
			if len(y) == 0:
				y = df.index
				n_points = len(y)
			for p in df.values:
				X.append(p)

	try:
		columns = ['id']
		df_out = pd.DataFrame(index=y)
		random_state = 0
		# Reduced number of epochs and sigma_iters. Previous were 1000 and 50.
		Y = tsne(np.array(X), perplexity=perplexity, n_epochs=10, sigma_iters=20, random_state=random_state, verbose=1)
		for t in range(n_revisions):
			df_out['t{}d0'.format(t)] = Y[t*n_points:(t+1)*n_points, 0]
			df_out['t{}d1'.format(t)] = Y[t*n_points:(t+1)*n_points, 1]

		if (len(df_out) - df_out.count()).sum() == 0:
			df_out.to_csv('./Output/{}-tsne_s4_{}p.csv'.format(dataset_dir.split('/')[-1], int(perplexity)), index_label='id')
			print(' '.join(sys.argv), 'OK')
		else:
			print(' '.join(sys.argv), 'crashed')
	except Exception as e:
		print(e)
		print(' '.join(sys.argv), 'crashed')

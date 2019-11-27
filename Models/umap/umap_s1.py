from Models import Shared
import umap
import sys
import glob
import natsort
import pandas as pd
import numpy as np
import random

## Usage: python umap_s1 <dataset_dir> <n_neigh>
## Ex: python umap_s1 ./Datasets/gaussians 15
if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    n_neigh = int(sys.argv[2])
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
    print(len(Xs))

    try:
        columns = ['id']
        df_out = pd.DataFrame(index=y)
        for t, X in enumerate(Xs):
            print(t, end=' ')
            sys.stdout.flush()
            random_state = random.randint(0, 100)
            Y = umap.UMAP(n_neighbors=n_neigh, n_components=2).fit_transform(X);
            df_out['t{}d0'.format(t)] = Y[:,0]
            df_out['t{}d1'.format(t)] = Y[:,1]
            # print(df_out.head())
        if (len(df_out) - df_out.count()).sum() == 0:
            df_out.to_csv('./Output/{}-umap_s1_{}p.csv'.format(dataset_dir.split('/')[-1], int(n_neigh)), index_label='id')
            print(' '.join(sys.argv), 'OK')
        else:
            print(' '.join(sys.argv), 'crashed')
    except:
        print(' '.join(sys.argv), 'crashed')

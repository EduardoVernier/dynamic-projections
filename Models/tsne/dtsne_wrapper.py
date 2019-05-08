from Models import Shared
import dynamic_tsne
from sklearn.utils import shuffle, check_random_state

import sys
import glob
import natsort
import pandas as pd
import numpy as np

## Usage: python dtsne_wrapper <dataset_dir> <perplexity> <lambda>
## Ex: python dtsne_wrapper ./Datasets/gaussians 30 0.1
if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    p = float(sys.argv[2])
    l = float(sys.argv[3])
    # print(dataset_dir, perplexity)

    Xs = []
    y = []
    n_revisions = 0
    if 'quickdraw' in dataset_dir or 'fashion' in dataset_dir:
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

    try:
        Ys = dynamic_tsne.dynamic_tsne(Xs, perplexity=p, lmbda=l,
                                       n_epochs=200, verbose=1, sigma_iters=50, random_state=0)

        df_out = pd.DataFrame(index=y)
        for t in range(len(Ys)):
            df_out['t{}d0'.format(t)] = Ys[t].T[0]
            df_out['t{}d1'.format(t)] = Ys[t].T[1]

        if (len(df_out) - df_out.count()).sum() == 0:
            df_out.to_csv('./Output/{}-dtsne_{}p_{}l.csv'.format(dataset_dir.split('/')[-1], int(p), float(l)), index_label='id')
            print(p, 'p', l, 'l', 'OK')
        else:
            print(p, 'p', l, 'l', 'crashed')
    except Exception as e:
        print(e)
        print(p, 'p', l, 'l', 'crashed')

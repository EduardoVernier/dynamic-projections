import numpy as np
import pandas as pd
import cv2
import glob
import random
import re
import math
import os
import sys
from sklearn.utils import shuffle
from keras import backend as K

# Load drawings from the quickdraw dataset
def load_drawings(base_path):
    categories = list(set([img.split('/')[-1].split('-')[0] for img in glob.glob(base_path + '*')]))
    X = []
    CATEGORIES = {}
    drawing_cat_str = []
    drawing_cat_id = []
    drawing_id = []
    drawing_t = []
    X_index = []

    print(sorted(categories))
    for cat_i, cat in enumerate(sorted(categories)):
        CATEGORIES[cat_i] = cat  # Maps a int to a string
        paths = glob.glob(base_path + cat + '-*')
        for p in paths:
            # Generate array from img
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            im = im / 255.
            X.extend(np.array([im]))

            # Drawing data
            drawing_cat_id.append(cat_i)
            drawing_cat_str.append(CATEGORIES[cat_i])
            match = re.match(r'.*/{}-(\d*)-(\d*).png'.format(CATEGORIES[cat_i]), p)
            drawing_id.append(int(match.group(1)))
            drawing_t.append(int(match.group(2)))
            X_index.append(len(X)-1)


    info_df = pd.DataFrame({'drawing_cat_id': drawing_cat_id,
                            'drawing_cat_str': drawing_cat_str,
                            'drawing_id': drawing_id,
                            't': drawing_t,
                            'X_index': X_index})



    # Replicate last image if drawing sequence doesn't have the maximum number of timesteps
    MAX_T = info_df['t'].max()
    g = info_df.groupby(by=['drawing_cat_id', 'drawing_id'])
    for key, df in g:
        max_row = (df[df['t'] == df['t'].max()])

        drawing_cat_str = []
        drawing_cat_id = []
        drawing_id = []
        drawing_t = []
        X_index = []

        for i in range(df['t'].max() + 1, MAX_T + 1):
            drawing_cat_id.append(int(max_row['drawing_cat_id'].values[0]))
            drawing_cat_str.append(max_row['drawing_cat_str'].values[0])
            drawing_id.append(int(max_row['drawing_id'].values[0]))
            X.extend([X[int(max_row['X_index'].values[0])]])
            X_index.append(len(X)-1)
            drawing_t.append(i)

        appendix = pd.DataFrame({'drawing_cat_id': drawing_cat_id,
                                 'drawing_cat_str': drawing_cat_str,
                                 'drawing_id': drawing_id,
                                 't': drawing_t,
                                 'X_index': X_index})
        info_df = info_df.append(appendix)

    X, info_df = shuffle(X, info_df, random_state=0)
    info_df['X_index'] = info_df.index
    info_df.index = range(len(info_df))
    info_df['X_index'] = range(len(info_df))
    info_df = info_df.astype({'drawing_cat_id':'int', 'drawing_id':'int', 't':'int'})
    return X, info_df, MAX_T+1, CATEGORIES

# def load_gaussians(dataset_path):
def load_tabular(dataset_path):
    X = []
    # meta info
    ts = []
    X_indices = []
    point_ids = []
    cats = []

    for csv in glob.glob(dataset_path + '*'):
        match = re.match(r'.*/.*-(\d*).csv', csv)
        t = int(match.group(1))
        df = pd.read_csv(csv, index_col=0)
        for id, row in df.iterrows():
            ts.append(t)
            point_ids.append(id)
            cats.append(id[0])
            X.append(row.values)
            X_indices.append(len(X)-1)

    info_df = pd.DataFrame({'point_id': point_ids,
                        't': ts,
                        'cat': cats,
                        'X_index': X_indices})
    X, info_df = shuffle(X, info_df, random_state=0)
    info_df['X_index'] = info_df.index
    info_df.index = range(len(info_df))
    info_df['X_index'] = range(len(info_df))
    return np.array(X), info_df, len(glob.glob(dataset_path + '*'))

# def save_quickdraw_vae_activations(encoder, X_flat, info_df, n_revisions, nb_name):
def save_drawing_vae_activations(encoder, X_flat, info_df, n_revisions, nb_name):
    # Collect all activations
    middle = len(encoder.layers) - 3
    middle_layer_output = K.function([encoder.layers[0].input],
                                     [encoder.layers[middle].output])
    layer_output = middle_layer_output([X_flat])[0]

    # Write activations to csv
    header = ['id']
    for t in range(n_revisions):
        for d in range(layer_output.shape[1]):
            header.append('t{}d{}'.format(t, d))

    csv_out = []
    gb = info_df.groupby(['drawing_cat_str', 'drawing_id'])
    for index, df in gb:  # Iterave over all drawing sequences
        drawing_id = index[0] + '-' + str(index[1])
        item_row = [drawing_id]
        for index, _ in df.sort_values('t').iterrows():  # For all timesteps
            for d in range(layer_output.shape[1]):  # Add all dimensions
                item_row.append(layer_output[index][d])
        csv_out.append(item_row)

    df_out = pd.DataFrame(csv_out, columns=header)
    df_out.to_csv('../../Output/{}.csv'.format(nb_name), index=False)
    return True


# def save_gaussian_vae_activations(encoder, X_flat, info_df, n_revisions, nb_name):
def save_tabular_vae_activations(encoder, X_flat, info_df, n_revisions, nb_name):
    # Collect all activations
    middle = len(encoder.layers) - 3
    middle_layer_output = K.function([encoder.layers[0].input],
                                     [encoder.layers[middle].output])
    layer_output = middle_layer_output([X_flat])[0]

    # Write activations to csv
    header = ['id']
    for t in range(n_revisions):
        for d in range(layer_output.shape[1]):
            header.append('t{}d{}'.format(t, d))

    csv_out = []
    gb = info_df.groupby(['point_id'])
    for index, df in gb:  # Iterave over all drawing sequences
        drawing_id = index[0]
        item_row = [drawing_id]
        for index, _ in df.sort_values('t').iterrows():  # For all timesteps
            for d in range(layer_output.shape[1]):  # Add all dimensions
                item_row.append(layer_output[index][d])
        csv_out.append(item_row)

    df_out = pd.DataFrame(csv_out, columns=header)
    df_out.to_csv('../../Output/{}.csv'.format(nb_name), index=False)
    return True


# def save_quickdraw_activations(ae, X_flat, info_df, n_revisions, nb_name):
def save_drawing_activations(ae, X_flat, info_df, n_revisions, nb_name):
	# Collect all activations
	middle = int(len(ae.layers)/2 - 1)
	middle_layer_output = K.function([ae.layers[0].input],
	                                 [ae.layers[middle].output])
	layer_output = middle_layer_output([X_flat])[0]

	# Write activations to csv
	header = ['id']
	for t in range(n_revisions):
	    for d in range(layer_output.shape[1]):
	        header.append('t{}d{}'.format(t, d))

	csv_out = []
	gb = info_df.groupby(['drawing_cat_str', 'drawing_id'])
	for index, df in gb:  # Iterave over all drawing sequences
	    drawing_id = index[0] + '-' + str(index[1])
	    item_row = [drawing_id]
	    for index, _ in df.sort_values('t').iterrows():  # For all timesteps
	        for d in range(layer_output.shape[1]):  # Add all dimensions
	            item_row.append(layer_output[index][d])
	    csv_out.append(item_row)

	df_out = pd.DataFrame(csv_out, columns=header)
	df_out.to_csv('../../Output/{}.csv'.format(nb_name), index=False)
	return True

def save_tsne_projection(info_df, n_revisions, nb_name):
    # Write csv headers
    header = ['id']
    for t in range(n_revisions):
        for d in range(len(info_df['projection'][0])):
            header.append('t{}d{}'.format(t, d))

    # Write data of each element.
    csv_out = []
    gb = info_df.groupby(['drawing_cat_str', 'drawing_id'])

    for index, df in gb:  # Iterave over all drawing sequences
        drawing_id = index[0] + '-' + str(index[1])
        item_row = [drawing_id]

        for index, _ in df.sort_values('t').iterrows():  # For all timesteps
            for d in range(len(info_df['projection'][0])):  # Add all dimensions
                item_row.append(info_df['projection'][index][d])

        csv_out.append(item_row)

    df_out = pd.DataFrame(csv_out, columns=header)
    df_out.to_csv('../../Output/{}.csv'.format(nb_name), index=False)

    return True

# def save_gaussian_activations(ae, X, info_df, n_revisions, nb_name):
def save_tabular_activations(ae, X, info_df, n_revisions, nb_name):
    # Collect all activations
    middle = int(len(ae.layers)/2 - 1)
    middle_layer_output = K.function([ae.layers[0].input],
                                     [ae.layers[middle].output])
    layer_output = middle_layer_output([X])[0]

    # Write activations to csv
    header = ['id']
    for t in range(n_revisions):
        for d in range(layer_output.shape[1]):
            header.append('t{}d{}'.format(t, d))

    csv_out = []
    gb = info_df.groupby(['point_id'])
    for index, df in gb:  # Iterave over all drawing sequences
        item_row = [index]
        for index, _ in df.sort_values('t').iterrows():  # For all timesteps
            for d in range(layer_output.shape[1]):  # Add all dimensions
                item_row.append(layer_output[index][d])
        csv_out.append(item_row)

    df_out = pd.DataFrame(csv_out, columns=header)
    df_out.to_csv('../../Output/{}.csv'.format(nb_name), index=False)
    return True

def tabular_to_dtsne_format(X, info_df):
    Xs = []
    y = []
    for i, df in info_df.groupby('t'):
        df = df.sort_values('point_id')
        if len(y) == 0:
            y = list(df['point_id'])
        Xs.append(X[df.index])
    return Xs, y

#!/bin/bash

datasets=$(cat Datasets/datasets.txt)
for d in $datasets; do 
  echo $d;
#  python Models/pca/pca_s1.py Datasets/$d;
#  python Models/pca/pca_s4.py Datasets/$d;
#  python Models/tsne/tsne_s1.py Datasets/$d 30;
#  python Models/tsne/tsne_s4.py Datasets/$d 30;
#  papermill Plots/trails-video.ipynb Plots/temp.ipynb --log-output -p dataset_id $d;
  papermill Plots/trails-image.ipynb Plots/temp.ipynb --log-output -p dataset_id $d;
done;


#papermill ./Metrics/template.ipynb ./Metrics/cartolastd.ipynb --log-output -p projection_paths 'Output/cartolastd-AE_10f_10f_2f_50ep.csv Output/cartolastd-VAE_10f_10f_2f_100ep.csv Output/cartolastd-tsne_s1_30p.csv Output/cartolastd-tsne_s4_30p.csv Output/cartolastd-dtsne_100p_0-1l.csv Output/cartolastd-pca_s1.csv Output/cartolastd-pca_s4.csv'
#papermill ./Metrics/template.ipynb ./Metrics/cifar10cnn.ipynb --log-output -p projection_paths 'Output/cifar10cnn-AE_10f_10f_2f_20ep.csv Output/cifar10cnn-VAE_100f_10f_2f_20ep.csv Output/cifar10cnn-tsne_s1_30p.csv Output/cifar10cnn-tsne_s4_30p.csv Output/cifar10cnn-dtsne_30p_0-1l.csv Output/cifar10cnn-pca_s1.csv Output/cifar10cnn-pca_s4.csv'
#papermill ./Metrics/template.ipynb ./Metrics/esc50.ipynb --log-output -p projection_paths 'Output/esc50-AE_10f_10f_2f_40ep.csv Output/esc50-VAE_100f_10f_2f_20ep.csv Output/esc50-tsne_s1_30p.csv Output/esc50-tsne_s4_30p.csv Output/esc50-dtsne_40p_0-05l.csv Output/esc50-pca_s1.csv Output/esc50-pca_s4.csv'
#papermill ./Metrics/template.ipynb ./Metrics/fashion.ipynb --log-output -p projection_paths 'Output/fashion-AE_784f_500f_500f_2000f_2f_40ep.csv Output/fashion-C2AE_32c_32c_32c_1568f_2f_40ep.csv Output/fashion-VAE_784f_2048f_1024f_512f_2f_0-25drop_20ep.csv Output/fashion-C2VAE_32c_64c_128c_2f_ep40.csv Output/fashion-tsne_s1_30p.csv Output/fashion-tsne_s4_30p.csv Output/fashion-dtsne_100p_0-1l.csv Output/fashion-pca_s1.csv Output/fashion-pca_s4.csv'
#papermill ./Metrics/template.ipynb ./Metrics/gaussians.ipynb --log-output -p projection_paths 'Output/gaussians-AE_10f_10f_2f_20ep.csv Output/gaussians-VAE_100f_10f_2f_20ep.csv Output/gaussians-tsne_s1_30p.csv Output/gaussians-tsne_s4_30p.csv Output/gaussians-dtsne_70p_0-1l.csv Output/gaussians-pca_s1.csv Output/gaussians-pca_s4.csv'
#papermill ./Metrics/template.ipynb ./Metrics/nnset.ipynb --log-output -p projection_paths 'Output/nnset-AE_10f_10f_2f_20ep.csv Output/nnset-VAE_100f_10f_2f_20ep.csv Output/nnset-tsne_s1_30p.csv Output/nnset-tsne_s4_30p.csv Output/nnset-dtsne_60p_0-01l.csv Output/nnset-pca_s1.csv Output/nnset-pca_s4.csv'
#papermill ./Metrics/template.ipynb ./Metrics/qtables.ipynb --log-output -p projection_paths 'Output/qtables-AE_10f_10f_2f_20ep.csv Output/qtables-VAE_100f_10f_2f_20ep.csv Output/qtables-tsne_s1_30p.csv Output/qtables-tsne_s4_30p.csv Output/qtables-dtsne_40p_0-05l.csv Output/qtables-pca_s1.csv Output/qtables-pca_s4.csv'
#papermill ./Metrics/template.ipynb ./Metrics/quickdraw.ipynb --log-output -p projection_paths 'Output/quickdraw-AE_784f_500f_500f_2000f_2f_20ep.csv Output/quickdraw-C2AE_32c_32c_32c_1568f_2f_2ep.csv Output/quickdraw-VAE_784f_2048f_1024f_512f_2f_0-25drop_10ep.csv Output/quickdraw-C2VAE_32c_64c_128c_6272f_2f_10ep.csv Output/quickdraw-tsne_s1_30p.csv Output/quickdraw-tsne_s4_30p.csv Output/quickdraw-dtsne_200p_0-1l.csv Output/quickdraw-pca_s1.csv Output/quickdraw-pca_s4.csv'
#papermill ./Metrics/template.ipynb ./Metrics/sorts.ipynb --log-output -p projection_paths 'Output/sorts-AE_10f_10f_2f_20ep.csv Output/sorts-VAE_100f_10f_2f_20ep.csv Output/sorts-tsne_s1_30p.csv Output/sorts-tsne_s4_30p.csv Output/sorts-dtsne_77p_0-01l.csv Output/sorts-pca_s1.csv Output/sorts-pca_s4.csv'
#papermill ./Metrics/template.ipynb ./Metrics/walk.ipynb --log-output -p projection_paths 'Output/walk-AE_10f_10f_2f_20ep.csv Output/walk-VAE_100f_10f_2f_20ep.csv Output/walk-tsne_s1_30p.csv Output/walk-tsne_s4_30p.csv Output/walk-dtsne_100p_0-01l.csv Output/walk-pca_s1.csv Output/walk-pca_s4.csv'

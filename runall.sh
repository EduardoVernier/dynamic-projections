#!/bin/bash

datasets=$(cat Datasets/datasets.txt)
for d in $datasets; do 
  echo $d;
#  python Models/pca/pca_s1.py Datasets/$d;
#  python Models/pca/pca_s4.py Datasets/$d;
#  python Models/tsne/tsne_s1.py Datasets/$d 30;
#  python Models/tsne/tsne_s4.py Datasets/$d 30;
  papermill Plots/trails-video.ipynb Plots/temp.ipynb --log-output -p dataset_id $d;
done;

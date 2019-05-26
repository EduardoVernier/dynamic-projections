# Dynamic projections

Set up virtual env and dependencies using pipenv.
https://pipenv.readthedocs.io/en/latest/
```
pip install pipenv
pipenv run pip install pip==18.0
pipenv install
sudo apt-get install python3-tk
```
To run a script use `pipenv run python <script_name>.py`. To open notebooks use `pipenv run jupyter notebook` or create a new shell with `pipenv shell` and then call `jupyter notebook`.

## Datasets ---  `./Datasets`

#### Dataset table
| dataset_id | n_items | n_dims | n_timesteps | n_classes | data_type          | source       | generator |
|------------|---------|--------|-------------|-----------|--------------------|--------------|-----------|
| gaussians  | 2000    | 100    | 10          | 10        | synthetic, tabular | dt-sne paper | -         |
|            |         |        |             |           |                    |              |           |
|            |         |        |             |           |                    |              |           |

#### Formatting

**Image datasets** -- The directory hierarchy doesn’t matter, all the metadata should be contained in the file name. `<class>-<id>-<time>.png`, e.g. `airplane-1234-10.png` -- 10th revision of airplane with id 1234.

**Tabular datasets** -- Each timestep is a single csv file named `<dataset_name>-<time>.csv`. The first column is the id and the next are the n features. I think this dtsne implementation only handles numerical features, so nothing categorical here for now.

**Output data (actual projections)** --- `./Output`

Single csv file with information about the model in the name in the format `<dataset>-<model_info>.csv`, as in `quickdraw-AE_728c_200c_p_d_200f_500f_2f.csv`. The previous string is an hypothetical filename for the results of projection using an AE with two convolutional layers of 728 and 200 kernels each, followed by max pooling and dropout layers and three dense layers of 200, 500 and 2 neurons each.

As for the contents of the file, the first column is the `id`, and the next are `t0d0, t0d1, ... t0dX, t1d0, ..., tTdX.` The number 't' is the timestep and 'd' is the representation dimension of each value.

## Generating the projections

##### Autoencoders ---  `./Models/ae`
The notebooks should contain information about training total time and performance metric (training/test accuracy and loss). The `Shared.py` file contains methods that might be useful for all notebooks and projection techniques e.g., saving projection, loading data.

##### Dynamic/static t-sne ---  `./Models/tsne`

From the root folder, we need to add the `tsne` folder to the PYTHONPATH and then run the dtsne_wrapper script.
```
export PYTHONPATH=${PYTHONPATH}:${PWD}/Models/tsne
python Models/tsne/dtsne_wrapper.py ./Datasets/gaussians 70 0.1
```
The default options are `n_epochs=200, sigma_iters=50`.

For static t-sne with strategies 1 and 4 (of the dt-sne paper):
```
export PYTHONPATH=${PYTHONPATH}:${PWD}/Models/tsne
python Models/tsne/tsne_s1.py ./Datasets/gaussians 70  # or
python Models/tsne/tsne_s4.py ./Datasets/gaussians 70
```

##### Principal component analysis ---  `./Models/pca`
```
export PYTHONPATH=${PYTHONPATH}:${PWD}/Models/
python Models/pca/pca_s1.py ./Datasets/gaussians  # or
python Models/pca/pca_s4.py ./Datasets/gaussians
```

## Visualizing the projections
There is a simple python tool based on matplotlib to quickly show and help us debug the generated projections. To use it, call
```
python Vis/Main.py ./Output/gaussians-pca_s4.csv ./Output/gaussians-AE_10f_2f_20ep.csv

python Vis/Main.py $(find Output/ -type f -name cartolastd*)
```

## Computing the metrics
The code for the metrics is located in a notebook called `template.ipynb`. For each dataset we use a tool called Papermill to instantiate a new notebook from the template. The two parameters that are needed are the output notebook path (remember to change name to dataset_id) and the list of output/projection files we want to analyse. This is the code that generates the analysis for the gaussians dataset:
```
papermill ./Metrics/template.ipynb ./Metrics/gaussians.ipynb --log-output -p projection_paths 'Output/gaussians-AE_10f_10f_2f_20ep.csv Output/gaussians-AE_10f_2f_20ep.csv Output/gaussians-tsne_s1_70p.csv Output/gaussians-tsne_s4_70p.csv Output/gaussians-dtsne_70p_0-1l.csv Output/gaussians-pca_s1.csv Output/gaussians-pca_s4.csv'

papermill ./Metrics/template.ipynb ./Metrics/quickdraw.ipynb --log-output -p projection_paths 'Output/quickdraw-AE_784f_500f_500f_2000f_2f_20ep.csv Output/quickdraw-C2AE_32c_32c_32c_1568f_2f_2ep.csv Output/quickdraw-VAE_784f_2048f_1024f_512f_2f_0-25drop_10ep.csv Output/quickdraw-C2VAE_32c_64c_128c_6272f_2f_10ep.csv Output/quickdraw-tsne_s1_200p.csv Output/quickdraw-dtsne_500p_0-5l.csv Output/quickdraw-pca_s1.csv Output/quickdraw-pca_s4.csv'

papermill ./Metrics/template.ipynb ./Metrics/cartola.ipynb --log-output -p projection_paths 'Output/cartolastd-AE_10f_2f_50ep.csv Output/cartolastd-AE_10f_10f_2f_50ep.csv Output/cartolastd-VAE_10f_2f_50ep.csv Output/cartolastd-tsne_s1_100p.csv Output/cartolastd-tsne_s4_1000p.csv Output/cartolastd-dtsne_100p_0-1l.csv Output/cartolastd-pca_s1.csv Output/cartolastd-pca_s4.csv'
```
The results are written in a csv file that goes into the `./Metrics/Results` directory.
To check the (tqdm) progress see the `log_<dataset_id>` file in real time.


## Plotting the results
TODO

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

## Data
### Input data ---  `./Datasets`

**Images** -- The directory hierarchy doesn’t matter, all the metadata should be contained in the file name. `<class>-<id>-<time>.png`, e.g. `airplane-1234-10.png` -- 10th revision of airplane with id 1234.

**Tabular data** -- Each timestep is a single csv file named `<dataset_name>-<time>.csv`. The first column is the id and the next are the n features. I think this dtsne implementation only handles numerical features, so nothing categorical here for now.

### Output data (actual projections) --- `./Output`

Single csv file with information about the model in the name in the format `<dataset>-<model_info>.csv`, as in `quickdraw-AE_728c_200c_p_d_200f_500f_2f.csv`. The previous string is an hypothetical filename for the results of projection using an AE with two convolutional layers of 728 and 200 kernels each, followed by max pooling and dropout layers and three dense layers of 200, 500 and 2 neurons each.

As for the contents of the file, the first column is the `id`, and the next are `t0d0, t0d1, ... t0dX, t1d0, ..., tTdX.` The number 't' is the timestep and 'd' is the representation dimension of each value.

### Metric Results --- `./MetricResults`
TODO (finish metric doc with new info)

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
`python Vis/Main.py ./Output/gaussians-pca_s4.csv ./Output/gaussians-AE_10f_2f_20ep.csv`

## Plotting the results
TODO

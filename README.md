# NeuTraj

This is a seed guided neural metric learning approach for calculating trajectory similarities.

## Require Packages
Pytorch, Numpy, trajectory_distance

## Running Procedures

### Create Folders
Before running the code, you need first create 3 empty folders:

*`data`: Place of the original data which is organized to a trajectory list. Each trajectory in it is a list of coordinate tuples (lon, lat).

*`features`: This folder contains the features that generated after the preprocessing.py. It contains four files: coor_seq, grid_seq, index_seq and seed_distance. 

*`model`: It is used for placing the NeuTraj model of each training epoch.

### Download Data
Due to the file limit of Github, we put the dataset on other sites. Please first download the data and put it in `data` folder. The toy dataset can be download at:  https://www.dropbox.com/s/ejoo1j21vjq7t7a/toy_trajs?dl=0

### Preprocessing
Run `preprocessing.py`. It filters the original data and maps the coordinates to grids. After such process, intermediate files which contain `coor_seq`, `grid_seq`, and `index_seq` are generated. Then, we calculate the pair-wise distance under the distance measure and get the `seed_distance`.

### Training & Evaluating
Run `train.py`. It trains NeuTraj under the supervision of seed distance. The parameters of NeuTraj can be modified in /tools/config.py

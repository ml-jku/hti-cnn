# Accurate Prediction of Biological Assays with High-throughput Microscopy Images and Convolutional Networks
This repository contains code to reproduce the results of "Accurate Prediction of Biological Assays with High-throughput Microscopy Images and Convolutional Networks".

# Dataset
The dataset used is based on the "Cell Painting Assay Dataset" (see https://github.com/gigascience/paper-bray2017 for download instructions). We also provide the subset of pre-processed images (in .npz format) used for our experiments here: https://ml.jku.at/software/cellpainting/dataset

# Instructions

## Configs
We use configuration files to set hyperparameters and directories, sample configurations are provided in the configs folder
and have to be adjusted accordingly.
Parameters from the configuration files can also be overwritten from the command line.

## Training
```
python main.py --config <configfile> --gpu <gpu-id> --j <number of dataloading threads> --training.batchsize <bs>
```

## Pre-trained Weights
Weights for the trained GapNet can be downloaded here: https://ml.jku.at/software/cellpainting/models/gapnet.pth.tar
When using the provided script specify the path to the downloaded weights via the --checkpoint switch, e.g.:
```
python main.py --config <configfile> --gpu <gpu-id> --checkpoint <path-to-checkpoint> --j <number of dataloading threads> --training.batchsize <bs>
```

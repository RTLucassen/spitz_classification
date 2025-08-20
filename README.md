# Artificial Intelligence-Based Classification of Spitz Tumors
This repository contains all code and trained model parameters to support the paper:  

***"Artificial Intelligence-Based Classification of Spitz Tumors"***  

The paper is currently under review.

## Contents
The repository contains several folders:
- `configs` contains two folders that include the configurations used for data preprocessing and model training.
- `evaluation` contains all python files that were used for evaluation of individual models and the ensemble, as well as the reader study. 
- `logistic regression` contains the python file for the logistic regression analysis.
- `models` contains the model implementation and trained parameters for HIPT and UNI.
  - Pretrained parameters for [HIPT](https://github.com/mahmoodlab/HIPT) and [UNI](https://github.com/mahmoodlab/UNI) can be downloaded from the original repositories.
  - Parameters for the Spitz classification trained as part of this project are available from the corresponding [HuggingFace repository](https://huggingface.co/RTLucassen/spitz_classification).
- `pipeline` contains all python files that were used for data transfer, de-identification, tissue segmentation, tessellation, and feature extraction. 
The implementation was designed to perform the preprocessing tasks in parallel, 
which may limit the generalizibility of the pipeline to different infrastructure and data storage systems.
- `simulation_experiment` contains the implementation of the simulation experiment.
- `start` contains all files to start the preprocessing tasks.
- `training` contains the implementation of the model training loop.

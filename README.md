# Artificial Intelligence-Based Classification of Spitz Tumors
This repository contains all code to support the paper:  

***"Artificial Intelligence-Based Classification of Spitz Tumors"***  

published in the Journal of Pathology Informatics and presented at ESMO AI & Digital Oncology 2025.

[[`arXiv`](https://arxiv.org/abs/2508.05391)] [[`JPI`](https://www.sciencedirect.com/science/article/pii/S2153353926001082)] [[`ESMO AI Poster`](https://github.com/RTLucassen/spitz_classification/blob/main/.github/ESMO_AI_poster.pdf)]

<div align="center">
  <img width="50%" alt="Simulation" src=".github\simulation.png">
</div>

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

## Citing
If you found our work useful in your research, please consider citing our paper:
```
@article{lucassen2026artificial,
title = {Artificial intelligence-based classification of {S}pitz tumors},
  journal = {Journal of Pathology Informatics},
  volume = {21},
  pages = {100650},
  year = {2026},
  doi = {https://doi.org/10.1016/j.jpi.2026.100650},
  author = {Ruben T. Lucassen and Marjanna Romers and Chiel F. Ebbelaar and Aia N. Najem and Donal P. Hayes and Antien L. Mooyaart and Sara Roshani and Liliane C.D. Wynaendts and Nikolas Stathonikos and Gerben E. Breimer and Anne M.L. Jansen and Mitko Veta and Willeke A.M. Blokx},
}
```

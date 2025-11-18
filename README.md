# Feature selection priors

The code uses tabularpriors data loaders, and TabPFN-wide's dataset extension approach.
Lasso is fit iteratively on the extended dataset to produce sparse coefficients that are 
gonna act as a ground truth label when pretraining a decoder. 

## Environment
Create and activate the environment:
```
uv tabprior
uv sync --frozen
source .tabprior/bin/activate
```

## Requirements
Install [tabularpriors](https://github.com/automl/tabularpriors/) and download [TabPFN-Wide](https://arxiv.org/abs/2510.06162) pretrained models.

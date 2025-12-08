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


Embedding layer -1 is correct, e.g. if we want embedding from layer 4, we need to pass 3 cause it starts enumerating with 0
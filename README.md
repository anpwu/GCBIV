# GCBIV

## Introduction
This repository contains the implementation code for paper:

**Generalized Instrumental Variable Learning** 

Anpeng Wu, Kun Kuang, Ruoxuan Xiong, and Fei Wu

## Env:

```shell
conda create -n tf-torch python=3.6
source activate tf-torch
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
conda install tensorflow-estimator==1.15.1 tensorflow-gpu==1.15.0
```

## Data Availability:
Data files are publicly available as follows: IHDP is available at http://www.fredjo.com/, Twins is available at http://www.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html. 
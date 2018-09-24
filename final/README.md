# 2018 Machine Learning Final Project
## Purpose

The purpose of the project is to choose the best options to complete the dialogue. 
The task can be seen in the data/testing_data.csv file. 

## Environment Setup
The project required following environment :
  - Python 3.5.2
  - CUDA 9.0.176
  - cuDNN 7.0.4

It's possible to run the project successfully only with the setup above. However, if an error occurred during execution, it is recommended to also apply the environment stated below. We have tested the project in the environment.
  - Ubuntu 16.04
  - NVIDIA Driver 384.130
  - Anaconda3-5.1.0
  - GPU - Tesla K80

## Installation
Simply clone this repo and cd to **final** folder.
```sh
$ git clone https://github.com/magonmonkey/ML2018SPRING.git
$ cd final
```
The files in **origin_files** are the data given in course. These data won't be used in demo. Instead, these data are processed by [naer_seg](https://github.com/naernlp/Segmentor) and stored in **data** folder. Please make sure the files in **data** are not modified. 

We also stored five models in **models** folder, if any of them are missing. You can download it from [here](https://goo.gl/koLMPd).
### Python package installation
These python packages are required in order to execute this project.

| Packages | Version |
| -------- | ------- |
| gensim | 3.4.0 |
| Keras | 2.2.0 |
| Keras-Applications | 1.0.2 |
| Keras-Preprocessing | 1.0.1 |
| numpy | 1.14.5 |
| tensorflow-gpu | 1.5.0 |

Or you could download all of them by 
```sh
(sudo) pip install -r requirements.txt 
```
## Execution
### Test
Execute the following command, the output file will be **final.csv**
(Please make sure you execute the command in **final** directory)
```bash
bash test.sh
```

### Train
Execute the following command, and the training process will begin. 
(Please make sure you execute the command in **final** directory)
```bash
bash train.sh
```

## (Optional) Chinese word segmentor
The files in **data** are processed by [Segmentor](https://github.com/naernlp/Segmentor). If you want to reproduce the result of the word segmentation. Please follow the instructions in https://github.com/naernlp/Segmentor to install the package. Then enter the command below and you will receive the same result as in **data**.
```sh
cd origin_files/
naer_seg *
mv testing_data.csv.seg testing_data.csv
```

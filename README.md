[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![GitHub pull requests](https://img.shields.io/github/issues-pr/likith012/mulEEG)
![GitHub issues](https://img.shields.io/github/issues/likith012/mulEEG)

# mulEEG: A Multi-View Representation Learning on EEG Signals
This repostiory contains code, results and dataset links for our paper titled ***mulEEG: A Multi-View Representation Learning on EEG Signals***. 📝
>**Authors:** <a name="myfootnote1"><sup>1</sup></a>Vamsi Kumar, <a name="myfootnote1"><sup>1</sup></a>Likith Reddy, Shivam Kumar Sharma, Kamalakar Dadi, Chiranjeevi Yarra, Bapi Raju and Srijithesh Rajendran

<sup>[1](#myfootnote1)</sup>Equal contribution

>**More details on the paper can be found [here](https://ieeexplore.ieee.org/document/9658706).**

## Table of contents
- [Introduction](https://github.com/likith012/IMLE-Net/edit/main/README.md#introduction-)
- [Highlights](https://github.com/likith012/IMLE-Net/edit/main/README.md#features-)
- [Results](https://github.com/likith012/IMLE-Net/edit/main/README.md#results)
- [Dataset](https://github.com/likith012/IMLE-Net/edit/main/README.md#organization-office)
- [Getting started](https://github.com/likith012/IMLE-Net/edit/main/README.md#getting-started-)
- [Getting the weights](https://github.com/likith012/IMLE-Net/edit/main/README.md#getting-the-weights-weight_lifting)
- [License and Citation](https://github.com/likith012/IMLE-Net/edit/main/README.md#license-and-citation-)

## Introduction 🔥

>Modeling effective representations using multiple views that
positively influence each other is challenging, and the existing methods perform poorly on Electroencephalogram (EEG) signals for sleep-staging tasks. In this paper, we propose a novel multi-view self-supervised method (mulEEG) for unsupervised EEG representation learning. Our method attempts to effectively utilize the complementary information available in multiple views to learn better representations. We introduce diverse loss that further encourages complementary information across multiple views. Our method with no access to labels, beats the supervised training  while outperforming multi-view baseline methods on transfer learning experiments carried out on sleep-staging tasks. We posit that our method was able to learn better representations by using complementary multi-views.

## Highlights ✨

- A model that learns patterns at the beat, rhythm, and channel level with high accuracy💯.
- An interpretable model that gives an explainability at the beat, rhythm and  channel level💥.
- Complete preprocessing pipeline, training and inference codes are available.
- Training weights are available to try out the model.

## Results :man_dancing:

> Performance metrics

|          | Macro ROC-AUC | Mean Accuracy | Max. F1-score |
| -------- | ------------- | ------------- | ------------- |
| Resnet101 | 0.8952 | 86.78 | 0.7558 |
| Mousavi et al.| 0.8654 | 84.19 | 0.7315 | 
| ECGNet | 0.9101 | 87.35 | 0.7712 |
| Rajpurkar et al. | 0.9155 | 87.91 | 0.7895 |
| **IMLE-Net**| **0.9216** | **88.85** | **0.8057** |



> Visualization of normalized attention scores with red having a higher attention score and yellow having a lower attention score for a 12-lead ECG signal.

<img src="/images/viz_nor_final.png" width="800">

> Channel Importance scores for the same 12-lead ECG signal.

<img src="/images/graph.png" width="400">


## Getting started 🥷
#### Setting up the environment
- All the development work is done using `Python 3.7`
- Install all the necessary dependencies using `requirements.txt` file. Run `pip install -r requirements.txt` in terminal
- Alternatively, set up the environment and train the model using the `Dockerfile`. Run `docker build -f Dockerfile -t <image_name> .`

#### What each file does

- `train.py` trains a particular model from scratch
- `preprocessing` contains the preprocessing scripts
- `models` contains scripts for each model
- `utils` contains utilities for `dataloader`, `callbacks` and `metrics`

#### Training the model
- All the models are implemented in `tensorflow` and `torch`
- Models implemented in `tensorflow` are `imle_net`, `mousavi` and `rajpurkar`
- Models implemented in `torch` are `ecgnet` and `resnet101`
- To log the training and validation metrics on wandb, set `--log_wandb` flag to `True`
- To train a particular model from scratch, `cd IMLE-Net`
- To run `tensorflow` models, `python train.py --data_dir data/ptb --model imle_net --batchsize 32 --epochs 60 --loggr True`
- To run `torch` models, `python torch_train.py --data_dir data/ptb --model ecgnet --batchsize 32 --epochs 60 --loggr True`

#### Testing the model
- To test the model, `cd IMLE-Net`
- To run `tensorflow` models, `python inference.py --data_dir data/ptb --model imle_net --batchsize 32`
- To run `torch` models, `python torch_inference.py --data_dir data/ptb --model ecgnet --batchsize 32`

#### Logs and checkpoints
- The logs are saved in `logs/` directory.
- The model checkpoints are saved in `checkpoints/` directory.

## Getting the weights :weight_lifting:

> Download the weights for several models trained on the PTB-XL dataset.

| Name | Model link |  
| ---- | ---------- |
| Mousavi et al.| [link](https://drive.google.com/file/d/13nUC_9mlSdw-I_HfFai4k8k9bgOruQ-x/view?usp=sharing) |
| ECGNet | [link](https://drive.google.com/file/d/1k0cgZBKQmkeVwu879NAtV-hDfLzCRCYJ/view?usp=sharing) |
| Rajpurkar et al. | [link](https://drive.google.com/file/d/18GZMDBAE2mHmQy8aXwD6cZoPjIAcavWX/view?usp=sharing) |
| **IMLE-Net**| [link](https://drive.google.com/file/d/1-ZJSEr_NtbLXWWx5otXT5ItE5p-Wc0HN/view?usp=sharing) |

## License and Citation 📰
The software is licensed under the Apache License 2.0. Please cite the following paper if you have used this code:
```
@INPROCEEDINGS{9658706,  
author={Reddy, Likith and Talwar, Vivek and Alle, Shanmukh and Bapi, Raju. S. and Priyakumar, U. Deva},  
booktitle={2021 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},   
title={IMLE-Net: An Interpretable Multi-level Multi-channel Model for ECG Classification},   
year={2021},  
pages={1068-1074}, 
doi={10.1109/SMC52423.2021.9658706}}
```


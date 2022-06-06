[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![GitHub pull requests](https://img.shields.io/github/issues-pr/likith012/mulEEG)
![GitHub issues](https://img.shields.io/github/issues/likith012/mulEEG)

# mulEEG: A Multi-View Representation Learning on EEG Signals
This repostiory contains code, results and dataset links for our MICCAI-2022 paper titled ***mulEEG: A Multi-View Representation Learning on EEG Signals***. 📝
>**Authors:** <a name="myfootnote1"><sup>1</sup></a>Vamsi Kumar, <a name="myfootnote1"><sup>1</sup></a>Likith Reddy, Shivam Kumar Sharma, Kamalakar Dadi, Chiranjeevi Yarra, Bapi Raju and Srijithesh Rajendran

<sup>[1](#myfootnote1)</sup>Equal contribution

>**More details on the paper can be found [here](https://arxiv.org/abs/2204.03272).**

## Table of contents
- [Introduction](https://github.com/likith012/mulEEG/edit/main/README.md#introduction-)
- [Highlights](https://github.com/likith012/mulEEG/edit/main/README.md#features-)
- [Results](https://github.com/likith012/mulEEG/edit/main/README.md#results)
- [Dataset](https://github.com/likith012/mulEEG/edit/main/README.md#organization-office)
- [Getting started](https://github.com/likith012/mulEEG/edit/main/README.md#getting-started-)
- [Getting the weights](https://github.com/likith012/mulEEG/edit/main/README.md#getting-the-weights-weight_lifting)
- [License and Citation](https://github.com/likith012/mulEEG/edit/main/README.md#license-and-citation-)

## Introduction 🔥

>Modeling effective representations using multiple views that
positively influence each other is challenging, and the existing methods perform poorly on Electroencephalogram (EEG) signals for sleep-staging tasks. In this paper, we propose a novel multi-view self-supervised method (mulEEG) for unsupervised EEG representation learning. Our method attempts to effectively utilize the complementary information available in multiple views to learn better representations. We introduce diverse loss that further encourages complementary information across multiple views. Our method with no access to labels, beats the supervised training  while outperforming multi-view baseline methods on transfer learning experiments carried out on sleep-staging tasks. We posit that our method was able to learn better representations by using complementary multi-views.

## Highlights ✨

- A self-supervised model pre-trained on unlabelled Electroencephalography (EEG) data beating the supervised counterpart 💥.
- Complete pre-processing pipeline, augmentation and training scripts are available for experimentation.
- Pre-trained model weights are provided for reproducability.

## Results :man_dancing:

> Linear evaluation results on Sleep-EDF dataset pre-trained on large SHHS dataset.

|          | Accuracy | κ | Macro F1-score |
| -------- | ------------- | ------------- | ------------- |
| Randomly Initialized | 38.68 | 0.1032 | 16.54 |
| Single-View | 76.73 | 0.6669 | 66.42 |
| Simple Fusion| 76.75 | 0.6658 | 65.78 | 
| CMC | 75.84 |  0.6520 | 64.40 |
| Supervised | 77.88 | 0.6838 | 67.84 |
| Ours | 78.18 | 0.6869 | 67.88 |
| **Ours + diverse loss**| **78.54** | **0.6914** | **68.10** |



> Our method performs better substantially in a low labelled data regime.

<img src="/images/semisupervised.png" width="600">

> t-SNE visualization using our method (no labels used) shows clear clusters and captures the sleep-staging progression observed clinically.

<img src="/images/cluster.png" width="750">


## Getting started 🥷
#### Setting up the environment
- All the development work is done using `Python 3.7`
- Install all the necessary dependencies using `requirements.txt` file. Run `pip install -r requirements.txt` in terminal
- Alternatively, set up the environment and train the model using the `Dockerfile`. Run `docker build -f Dockerfile -t <image_name> .`

#### What each file does
TODO
#### Training the model
TODO
#### Testing the model
TODO

#### Logs and checkpoints
- The logs are saved in `logs/` directory.
- The model checkpoints are saved in `checkpoints/` directory.

## Getting the weights :weight_lifting:

> Download the model weights for all the baseline methods and ours.

| Name | Sleep-EDF | SHHS |  
| ---- | ---------- | ---- |
| Single-View | [link](/weights/sleepedf/single_view.pt) | [link](/weights/shhs/single_view.pt) |
| Simple Fusion| [link](/weights/sleepedf/simple_fusion.pt) | [link](/weights/shhs/simple_fusion.pt) |
| CMC | [link](/weights/sleepedf/cmc.pt) | [link](/weights/shhs/cmc.pt) |
| Supervised | [link](/weights/sleepedf/supervised_pretrain.pt) | [link](/weights/shhs/supervised_pretrain.pt) |
| **Ours + diverse loss**| [link](/weights/sleepedf/ours_diverse.pt) | [link](/weights/shhs/ours_diverse.pt) |



## License and Citation 📰
The software is licensed under the Apache License 2.0. Please cite the following paper if you have used this code:
```
@misc{kumar2022muleeg,
    title={mulEEG: A Multi-View Representation Learning on EEG Signals},
    author={Vamsi Kumar and Likith Reddy and Shivam Kumar Sharma and Kamalakar Dadi and Chiranjeevi Yarra and Bapi S. Raju and Srijithesh Rajendran},
    year={2022},
    eprint={2204.03272},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


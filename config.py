"""Configuration file for the project."""

__author__ = "Likith Reddy, Vamsi Kumar"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com, vamsi81523@gmail.com"


import os
import torch


class Config(object):
    def __init__(self, wandb=None) -> None:

        # path
        self.src_path = os.path.join(os.getcwd(), "data")
        self.wandb = wandb

        # augmentation
        self.degree = 0.05
        self.mask_max_points = 200
        self.mask_min_points = 50

        # time domain
        self.tc_hidden_dim = 128
        self.input_channels = 1

        # loss
        self.temperature = 1
        self.intra_temperature = 10
        self.use_cosine_similarity = True

        # optimizer
        self.optimizer = "adam"
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.0003

        # training and evaluation
        self.num_epoch = 200
        self.batch_size = 256
        self.num_ft_epoch = 100
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.drop_last = True

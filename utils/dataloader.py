"""Augmentations pipeline for self-supervised pre-training. 

Two variations of augmentations are used. Jittering, where random uniform noise is added to the EEG signal depending on its peak-to-peak values, 
along with masking, where signals are masked randomly. Flipping, where the EEG signal is horizontally flipped randomly, and scaling, where EEG 
signal is scaled with Gaussian noise.

This file can also be imported as a module and contains the following:

    * Load_Dataset - Loads the dataset and applies the augmentations.
    * data_generator - Generates a dataloader for the dataset.
    * cross_data_generator - Generates a k-fold dataloader for the given dataset. 
"""
__author__ = "Likith Reddy, Vamsi Kumar"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com, vamsi81523@gmail.com"

from typing import Type, Tuple, Dict, Union
import os

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utils.augmentations import augment
from config import Config


class Load_Dataset(Dataset):
    """ Class to load the dataset and applies the augmentations.

    Attributes
    ----------
    dataset: Dict[str, np.ndarray]
        Contains the EEG data and labels.
    config: Type[Config]
        Configuration object specifying the model hyperparameters.
    training_mode: str, optional
        Defines the training mode. (default: 'self_supervised')

    Methods
    -------
    __getitem__(index)
        Returns indexed data and labels depending on the training mode.

    __len__()
        Returns the length of the dataset.

    """

    def __init__(
        self,
        dataset: Dict[str, torch.Tensor],
        config: Type[Config],
        training_mode: str = "self_supervised",
    ) -> None:
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"][:, 0, :].unsqueeze(1)
        y_train = dataset["labels"]

        # checking the shape of the data
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # make sure the channels are in second dim
        if X_train.shape.index(min(X_train.shape)) != 1:
            X_train = X_train.permute(0, 2, 1)

        # checking for the instance of the dataset
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
        else:
            self.x_data = X_train

        if isinstance(y_train, np.ndarray):
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.y_data = y_train

        self.len = X_train.shape[0]
        self.config = config
        self.training_mode = training_mode

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns indexed data and labels depending on the training mode."""

        if self.training_mode == "self_supervised":
            # get the augmented data for the pretraining
            weak_data, strong_data = augment(self.x_data[index], self.config)

            if isinstance(weak_data, np.ndarray):
                weak_data = torch.from_numpy(weak_data)
            if isinstance(strong_data, np.ndarray):
                strong_data = torch.from_numpy(strong_data)

            return weak_data, strong_data

        else:
            return self.x_data[index].float(), self.y_data[index]

    def __len__(self) -> int:
        """ Returns the length of the dataset."""

        return self.len


def data_generator(data_path: str, config: Type[Config]) -> Type[DataLoader]:

    """ Generates a dataloader for the dataset.

    Parameters
    ----------
    data_path: str
        A string indicating the path to the dataset.
    config: Config Object
        Configuration object specifying the model hyperparameters.

    Returns
    -------
    Type[DataLoader]
        DataLoader object for the given dataset.
    """

    train_dataset = torch.load(os.path.join(data_path, "pretext.pt"))

    train_dataset = Load_Dataset(train_dataset, config)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=config.drop_last,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader


def cross_data_generator(
    data_path: str, train_idxs: list, val_idxs: list, config: Type[Config]
) -> Union[Tuple[Type[DataLoader], ...], int]:
    """ Generates a k-fold dataloader for the given dataset. 

    Here the splits are made depending on the subject id, to make sure that there is no data leakage.

    Parameters
    ----------
    data_path: str
        Data path to the dataset.
    train_idxs: list
       Contains the indices of the training data.
    val_idxs: list
        Contains the indices of the validation data.
    config: Type[Config]
        An object containing the hyperparameters.

    Returns
    -------
    Union[Tuple[Type[DataLoader], ...], int]
        DataLoader object for the training and validation data.

    """

    train_ds = torch.load(os.path.join(data_path, "train.pt"))
    train_ds["samples"] = train_ds["samples"][:, 0, :].unsqueeze(1)
    valid_ds = torch.load(os.path.join(data_path, "val.pt"))
    valid_ds["samples"] = valid_ds["samples"][:, 0, :].unsqueeze(1)

    """
    train_subs : list of subjects for training; this list depends on the seed set during the preprocessing
        with the default seed that we are using these are the subjects present in the training split

    train_segs : list of number of epochs of EEG data present in each subject of the train_subs list  
    """

    train_subs = [48, 72, 24, 30, 34, 50, 38, 15, 60, 12]
    train_segs = [3937, 2161, 3448, 1783, 3083, 2429, 3647, 2714, 3392, 2029]

    """
    val_subs : list of subjects for validation; this list depends on the seed set during the preprocessing

    val_segs : list of number of epochs of EEG data present in each subject of the val_subs list
    """

    val_subs = [23, 26, 37, 44, 49, 51, 54, 59, 73, 82]
    val_segs = [2633, 2577, 2427, 2287, 2141, 2041, 2864, 3071, 4985, 3070]

    segs = train_segs + val_segs

    if train_idxs != []:

        dataset = {}
        train_dataset = {}
        valid_dataset = {}

        # combine both training and validation data
        dataset["samples"] = torch.from_numpy(
            np.vstack((train_ds["samples"], valid_ds["samples"]))
        )
        dataset["labels"] = torch.from_numpy(
            np.hstack((train_ds["labels"], valid_ds["labels"]))
        )

        # split the data depending on the subject id
        dataset["samples"] = torch.split(dataset["samples"], segs)
        dataset["labels"] = torch.split(dataset["labels"], segs)
        print("Split Shape", len(dataset["samples"]))

        # create split for training
        train_dataset["samples"] = [dataset["samples"][i] for i in train_idxs]
        train_dataset["labels"] = [dataset["labels"][i] for i in train_idxs]

        train_dataset["samples"] = torch.cat(train_dataset["samples"])
        train_dataset["labels"] = torch.cat(train_dataset["labels"])
        train_dataset = Load_Dataset(train_dataset, config, training_mode="ft")

        # create split for validation
        valid_dataset["samples"] = [dataset["samples"][i] for i in val_idxs]
        valid_dataset["labels"] = [dataset["labels"][i] for i in val_idxs]

        valid_dataset["samples"] = torch.cat(valid_dataset["samples"])
        valid_dataset["labels"] = torch.cat(valid_dataset["labels"])
        valid_dataset = Load_Dataset(valid_dataset, config, training_mode="ft")

        # create training dataloader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=config.drop_last,
            num_workers=10,
            pin_memory=True,
            persistent_workers=True,
        )

        # create validation dataloader
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=config.drop_last,
            num_workers=10,
            pin_memory=True,
            persistent_workers=True,
        )

        del dataset
        del train_dataset
        del valid_dataset

        return train_loader, valid_loader

    ret = len(val_subs) + len(train_subs)
    del train_ds
    del valid_ds

    return ret

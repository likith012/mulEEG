"""Generates data splits for sleep-edf dataset. 

    This file is used as script and contains the following functions:

    * gen_sleepedf - Generates the pretext, train and test splits on sleep-edf dataset for self-supervised pre-training.

"""

__author__ = "Likith Reddy, Vamsi Kumar"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com, vamsi81523@gmail.com"


import os
import torch
import numpy as np


def gen_sleepedf(files:np.ndarray,save_path:str):

    """
    Generates the pretext, train and test splits on sleep-edf dataset for self-supervised pre-training.

    Attributes:
        files: list
            List of files to be used for generating the splits.
        save_path: str
            Path to save the splits.
    """

    ######## Pretext files########
    pretext_files = list(np.random.choice(files,58,replace=False))
    print("pretext files: ", len(pretext_files))

    # load files
    X = np.load(pretext_files[0])["x"]
    y = np.load(pretext_files[0])["y"]

    for i, file in enumerate(pretext_files[1:]):
        print(os.path.basename(file))
        X = np.vstack((X, np.load(file)["x"]))
        y = np.append(y, np.load(file)["y"])

    data_save = dict()
    data_save["samples"] = torch.from_numpy(X.transpose(0, 2, 1))
    data_save["labels"] = torch.from_numpy(y)

    torch.save(data_save, os.path.join(save_path, "pretext.pt"))

    ######## Training files ##########
    training_files = list(np.random.choice(sorted(list(set(files) - set(pretext_files))), 10, replace=False))

    print("\n ============================================== \n")
    print("training files: ", len(training_files))

    # load files
    X = np.load(training_files[0])["x"]
    y = np.load(training_files[0])["y"]

    for file in training_files[1:]:
        print(os.path.basename(file))
        X = np.vstack((X, np.load(file)["x"]))
        y = np.append(y, np.load(file)["y"])

    data_save = dict()
    data_save["samples"] = torch.from_numpy(X.transpose(0, 2, 1))
    data_save["labels"] = torch.from_numpy(y)
    torch.save(data_save, os.path.join(save_path, "train.pt"))

    ######## Test files ##########
    test_files = sorted(list(set(files) - set(pretext_files) - set(training_files)))

    print("\n =========================================== \n")
    print("test files: ", len(test_files))

    # load files
    X = np.load(test_files[0])["x"]
    y = np.load(test_files[0])["y"]

    for file in test_files[1:]:
        print(os.path.basename(file))
        X = np.vstack((X, np.load(file)["x"]))
        y = np.append(y, np.load(file)["y"])

    data_save = dict()
    data_save["samples"] = torch.from_numpy(X.transpose(0, 2, 1))
    data_save["labels"] = torch.from_numpy(y)

    torch.save(data_save, os.path.join(save_path, "test.pt"))
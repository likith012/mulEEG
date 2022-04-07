"""Generates data splits for sleep-edf dataset. 

This file is used as script and contains the following functions:

    * main - Generates the pretext, train and test splits on sleep-edf dataset for self-supervised pre-training.

"""

__author__ = "Likith Reddy, Vamsi Kumar"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com, vamsi81523@gmail.com"


import os
import torch
import numpy as np
import argparse



def main():
    """Generates the pretext, train and test splits on sleep-edf dataset for self-supervised pre-training."""

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

    torch.save(data_save, os.path.join(output_dir, "pretext.pt"))

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
    torch.save(data_save, os.path.join(output_dir, "train.pt"))

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

    torch.save(data_save, os.path.join(output_dir, "test.pt"))

if __name__ == "__main__":
    # Seed for reproducibility
    seed = 1234
    np.random.seed(seed)

    current_path = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=os.path.join(current_path, 'data'), help="File path to the PSG and annotation files.")
    args = parser.parse_args()

    data_dir = os.path.join(args.dir, 'numpy_subjects')
    output_dir = os.path.join(args.dir, 'less_subjs')
    files = os.listdir(data_dir)
    files = np.array([os.path.join(data_dir, i) for i in files])
    files.sort()

    main()
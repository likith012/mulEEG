"""Generates data splits for shhs dataset. 

    This file is used as script and contains the following functions:

    * gen_shhs - Generates the pretext, train and test splits on shhs dataset for self-supervised pre-training.

"""

__author__ = "Likith Reddy, Vamsi Kumar"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com, vamsi81523@gmail.com"

import numpy as np
import torch
import os

def gen_shhs(files:np.ndarray,save_path:str):

    """
    Generates the pretext, train and test splits on shhs dataset for self-supervised pre-training.

    Attributes:
        files: list
            List of files to be used for generating the splits.
        save_path: str
            Path to save the splits. 
    
    """

    ######## pretext files##########
    pretext_files = list(np.random.choice(files,264,replace=False))    #change

    print("pretext files: ", len(pretext_files))

    ### Below code is for making pretext.pt train.pt val.pt

    X_train = np.load(pretext_files[0])["x"]
    y_train = np.load(pretext_files[0])["y"]
    c=0
    for np_file in pretext_files[1:]:
       print(os.path.basename(np_file))
       x_dat = np.load(np_file)["x"]
       if x_dat.shape[-1]==2:
           X_train = np.vstack((X_train,x_dat))
           y_train = np.append(y_train, np.load(np_file)["y"])
       else:
           print('Deleted')


    data_save = dict()
    data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
    data_save["labels"] = torch.from_numpy(y_train)

    torch.save(data_save, os.path.join(save_path, "pretext.pt"))

    ######## training files ##########
    training_files = list(np.random.choice(sorted(list(set(files)-set(pretext_files))),31,replace=False))  #change

    print("\n =========================================== \n")
    print("training files: ", len(training_files))

    # load files
    X_train = np.load(training_files[0])["x"]
    y_train = np.load(training_files[0])["y"]

    for np_file in training_files[1:]:
       x_dat = np.load(np_file)["x"]
       print(os.path.basename(np_file),x_dat.shape)
       if x_dat.shape[-1]==2:
           X_train = np.vstack((X_train,x_dat))
           y_train = np.append(y_train, np.load(np_file)["y"])
       else:
           print('Deleted')

    data_save = dict()
    data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
    data_save["labels"] = torch.from_numpy(y_train)
    torch.save(data_save, os.path.join(save_path, "train.pt"))

    ######## validation files ##########
    validation_files = sorted(list(set(files)-set(pretext_files)-set(training_files))) #list(np.random.choice(sorted(list(set(files)-set(pretext_files)-set(training_files))),32,replace=False))    # left =32

    print("\n =========================================== \n")
    print("validation files: ", len(validation_files))

    # load files
    X_train = np.load(validation_files[0])["x"]
    y_train = np.load(validation_files[0])["y"]

    for np_file in validation_files[1:]:
       print(os.path.basename(np_file))
       x_dat = np.load(np_file)["x"]
       if x_dat.shape[-1]==2:
           X_train = np.vstack((X_train,x_dat))
           y_train = np.append(y_train, np.load(np_file)["y"])
       else:
           print('Deleted')


    data_save = dict()
    data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
    data_save["labels"] = torch.from_numpy(y_train)

    torch.save(data_save, os.path.join(save_path, "val.pt"))
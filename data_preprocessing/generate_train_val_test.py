
import os
import torch
import numpy as np
import argparse

seed = 1234
np.random.seed(seed)


parser = argparse.ArgumentParser()

parser.add_argument("--dir", type=str, default="/scratch/SLEEP_data",
                    help="File path to the PSG and annotation files.")

args = parser.parse_args()


data_dir = os.path.join(args.dir, "numpy_subjects")
output_dir = args.dir+"/less_subjs/"

files = os.listdir(data_dir)
files = np.array([os.path.join(data_dir, i) for i in files])
files.sort()



######## pretext files##########
pretext_files = list(np.random.choice(files,58,replace=False))

print("pretext files: ", len(pretext_files))

# load files

X_train = np.load(pretext_files[0])["x"]
y_train = np.load(pretext_files[0])["y"]

for i,np_file in enumerate(pretext_files[1:]):
    print(os.path.basename(np_file))
    X_train = np.vstack((X_train, np.load(np_file)["x"]))
    y_train = np.append(y_train, np.load(np_file)["y"])
    if i==19:
        break


data_save = dict()
data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_train)

torch.save(data_save, os.path.join(output_dir, "pretext.pt"))

######## training files ##########
training_files = list(np.random.choice(sorted(list(set(files)-set(pretext_files))),10,replace=False))

print("\n =========================================== \n")
print("training files: ", len(training_files))

# load files
X_train = np.load(training_files[0])["x"]
y_train = np.load(training_files[0])["y"]

for np_file in training_files[1:]:
    print(os.path.basename(np_file))
    X_train = np.vstack((X_train, np.load(np_file)["x"]))
    y_train = np.append(y_train, np.load(np_file)["y"])

data_save = dict()
data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_train)
torch.save(data_save, os.path.join(output_dir, "train.pt"))

######## validation files ##########
validation_files = sorted(list(set(files)-set(pretext_files)-set(training_files)))

print("\n =========================================== \n")
print("validation files: ", len(validation_files))

# load files
X_train = np.load(validation_files[0])["x"]
y_train = np.load(validation_files[0])["y"]

for np_file in validation_files[1:]:
    print(os.path.basename(np_file))
    X_train = np.vstack((X_train, np.load(np_file)["x"]))
    y_train = np.append(y_train, np.load(np_file)["y"])


data_save = dict()
data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_train)

torch.save(data_save, os.path.join(output_dir, "val.pt"))

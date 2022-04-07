import torch
import numpy as np
import os
from torch.utils.data import Dataset
from utils.augmentations import augment


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(
        self, dataset, config, data_path, training_mode="self_supervised", wh="train"
    ):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"][:, 0, :].unsqueeze(1)
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if (
            X_train.shape.index(min(X_train.shape)) != 1
        ):  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

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

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            # return torch.rand(1,3000),torch.rand(1,3000)
            weak_dat, strong_dat = augment(self.x_data[index], self.config)

            if isinstance(weak_dat, np.ndarray):
                weak_dat = torch.from_numpy(weak_dat)

            if isinstance(strong_dat, np.ndarray):
                strong_dat = torch.from_numpy(strong_dat)

            return weak_dat, strong_dat
            # return weak_dat,torch.rand(6,60),strong_dat,torch.rand(6,60)

        else:
            return self.x_data[index].float(), self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs):

    train_dataset = torch.load(os.path.join(data_path, "pretext.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, data_path=data_path)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        drop_last=configs.drop_last,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader


def ft_data_generator(data_path, configs):
    train_ds = torch.load(os.path.join(data_path, "train.pt"))
    valid_ds = torch.load(os.path.join(data_path, "val.pt"))

    test_ds = Load_Dataset(
        valid_ds, configs, data_path=data_path, training_mode="ft", wh="valid"
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=configs.batch_size,
        shuffle=False,
        drop_last=configs.drop_last,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )

    train_ds = Load_Dataset(
        train_ds, configs, data_path=data_path, training_mode="ft", wh="train"
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=configs.batch_size,
        shuffle=True,
        drop_last=configs.drop_last,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, test_loader


def cross_data_generator(data_path, train_idxs, val_idxs, configs):
    train_ds = torch.load(os.path.join(data_path, "train.pt"))
    train_ds["samples"] = train_ds["samples"][:, 0, :].unsqueeze(1)
    valid_ds = torch.load(os.path.join(data_path, "val.pt"))
    valid_ds["samples"] = valid_ds["samples"][:, 0, :].unsqueeze(1)

    train_subs = [48, 72, 24, 30, 34, 50, 38, 15, 60, 12]
    train_segs = [3937, 2161, 3448, 1783, 3083, 2429, 3647, 2714, 3392, 2029]

    val_subs = [23, 26, 37, 44, 49, 51, 54, 59, 73, 82]
    val_segs = [2633, 2577, 2427, 2287, 2141, 2041, 2864, 3071, 4985, 3070]

    segs = train_segs + val_segs

    if train_idxs != []:

        dataset = {}
        train_dataset = {}
        valid_dataset = {}
        dataset["samples"] = torch.from_numpy(
            np.vstack((train_ds["samples"], valid_ds["samples"]))
        )
        dataset["labels"] = torch.from_numpy(
            np.hstack((train_ds["labels"], valid_ds["labels"]))
        )

        dataset["samples"] = torch.split(dataset["samples"], segs)
        dataset["labels"] = torch.split(dataset["labels"], segs)
        print("Split Shape", len(dataset["samples"]))

        train_dataset["samples"] = [dataset["samples"][i] for i in train_idxs]
        train_dataset["labels"] = [dataset["labels"][i] for i in train_idxs]

        train_dataset["samples"] = torch.cat(train_dataset["samples"])
        train_dataset["labels"] = torch.cat(train_dataset["labels"])
        print(
            "Train Shape", train_dataset["samples"].shape, train_dataset["labels"].shape
        )
        train_dataset = Load_Dataset(
            train_dataset, configs, data_path=data_path, training_mode="ft"
        )

        valid_dataset["samples"] = [dataset["samples"][i] for i in val_idxs]
        valid_dataset["labels"] = [dataset["labels"][i] for i in val_idxs]

        valid_dataset["samples"] = torch.cat(valid_dataset["samples"])
        valid_dataset["labels"] = torch.cat(valid_dataset["labels"])
        valid_dataset = Load_Dataset(
            valid_dataset, configs, data_path=data_path, training_mode="ft"
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=configs.batch_size,
            shuffle=True,
            drop_last=configs.drop_last,
            num_workers=10,
            pin_memory=True,
            persistent_workers=True,
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=configs.batch_size,
            shuffle=False,
            drop_last=configs.drop_last,
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

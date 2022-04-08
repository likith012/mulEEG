


import argparse
import glob
import math
import ntpath
import os
import shutil


from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter,lfilter
from mne.io import concatenate_raws, read_raw_edf
from generate_sleepedf import gen_sleepedf

import dhedfreader

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30


def combine_to_subjects(root_dir, output_dir):
    sampling_rate = 100.
    files = os.listdir(root_dir)
    files = [os.path.join(root_dir, i) for i in files]

    files_dict = {}

    for i in files:
        file_name = i.split(os.sep)[-1]
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)

    os.makedirs(output_dir, exist_ok=True)
    for i in files_dict:
        if len(files_dict[i]) == 2:
            x1 = np.load(files_dict[i][0])["x"]
            x2 = np.load(files_dict[i][1])["x"]
            new_x = np.concatenate((x1, x2), axis=0)

            y1 = np.load(files_dict[i][0])["y"].tolist()
            y2 = np.load(files_dict[i][1])["y"].tolist()
            y1.extend(y2)
            y1 = np.array(y1)
        else:
            new_x = np.load(files_dict[i][0])["x"]
            y1 = np.load(files_dict[i][0])["y"]

        # Saving as numpy files
        # print(file, x_values.shape[0], y_values.shape[0])
        filename = "subject_" + str(i) + ".npz"

        save_dict = {
            "x": new_x,
            "y": y1,
            "fs": sampling_rate
        }
        np.savez(os.path.join(output_dir, filename), **save_dict)
        print(" ---------- Combining files to subjects is done ---------")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./SLEEP_data",
                        help="File path to the PSG and annotation files.")

    parser.add_argument("--save_path", type=str, default="./SLEEP_data",
                        help="Path to save preprocess files")


    args = parser.parse_args()

    channels = 1


    data_dir = os.path.join(args.data_dir,"/physionet-sleep-data/")
    subjects_output_dir = os.path.join(args.save_path,"/numpy_subjects/")
    output_dir = os.path.join(args.save_path,"/numpy_saves/")
    save_dir = os.path.join(args.save_path,"/data/")

    print(data_dir)

    # Output dir
    if not os.path.exists(subjects_output_dir):
        os.makedirs(subjects_output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Select channel
    all_picks = ['EEG Fpz-Cz',
            'EEG Pz-Oz',
            'EOG horizontal',
            'Resp oro-nasal',
            'EMG submental',
            'Temp rectal',
            'Event marker']
    select_ch = all_picks[:channels]

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(data_dir, "*Hypnogram.edf"))

    psg_fnames.sort()
    ann_fnames.sort()

    print("Number of PSG files: ", len(psg_fnames))
    print("Number of annotation files: ", len(ann_fnames))

    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):

        print(i)

        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)
        sampling_rate = raw.info['sfreq']
        raw_ch_df = raw.to_data_frame()[select_ch]

        raw_ch_df.set_index(np.arange(len(raw_ch_df)))

        # Get raw header
        f = open(psg_fnames[i], 'r', errors='ignore')
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()
        raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

        # Read annotation and its header
        f = open(ann_fnames[i], 'r', errors='ignore')
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = zip(*reader_ann.records())
        f.close()
        ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")

        # Assert that raw and annotation files start at the same time
        assert raw_start_dt == ann_start_dt

        # Generate label and remove indices
        remove_idx = []    # indicies of the data that will be removed
        labels = []        # indicies of the data that have labels
        label_idx = []

        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            label = ann2label[ann_str[2:-1]]
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=np.int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
                label_idx.append(idx)

            else:
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
                remove_idx.append(idx)

        labels = np.hstack(labels)

        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))

        # Select only the data with labels
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)

        # Remove extra index
        if len(label_idx) > len(select_idx):
            extra_idx = np.setdiff1d(label_idx, select_idx)
            # Trim the tail
            if np.all(extra_idx > select_idx[-1]):

                n_label_trims = int(math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate)))
                if n_label_trims!=0:

                    labels = labels[:-n_label_trims]

        # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values[select_idx]

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y)

        # Select on sleep periods
        w_edge_mins = 30

        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)

        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1

        select_idx = np.arange(start_idx, end_idx+1)
        print("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        print("Data after selection: {}, {}".format(x.shape, y.shape))

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")


        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "header_raw": h_raw,
            "header_annotation": h_ann,
        }
        np.savez(os.path.join(output_dir, filename), **save_dict)

        print ("\n====================================================================================\n")

    combine_to_subjects(output_dir, subjects_output_dir)

    files = os.listdir(subjects_output_dir)
    files = np.array([os.path.join(subjects_output_dir, i) for i in files])
    files.sort()

    # generates the final preprocessed data
    gen_sleepedf(files, subjects_output_dir)

if __name__ == "__main__":

    seed = 1234
    np.random.seed(seed)

    main()
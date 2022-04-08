#!/usr/bin/env python
# coding: utf-8

import re, datetime, operator
import numpy as np
import os
import argparse
import warnings
import pandas as pd
import xml.etree.ElementTree as ET
import torch

from datetime import datetime
from collections import namedtuple
from mne.io import read_raw_edf

warnings.filterwarnings("ignore")

EVENT_CHANNEL = 'EDF Annotations'

class EDFEndOfData(Exception): pass

def tal(tal_str):
    '''Return a list with (onset, duration, annotation) tuples for an EDF+ TAL
  stream.
  '''
    exp = '(?P<onset>[+\-]\d+(?:\.\d*)?)' + '(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + '(\x14(?P<annotation>[^\x00]*))?' + '(?:\x14\x00)'

    def annotation_to_list(annotation):
        return str(annotation.encode('utf-8')).split('\x14') if annotation else []

    def parse(dic):
        return (
      float(dic['onset']),
      float(dic['duration']) if dic['duration'] else 0.,
      annotation_to_list(dic['annotation']))

    return [parse(m.groupdict()) for m in re.finditer(exp, tal_str)]


def edf_header(f):
    h = {}
    assert f.tell() == 0  # check file position
    assert f.read(8) == '0       '

    # recording info)
    h['local_subject_id'] = f.read(80).strip()
    h['local_recording_id'] = f.read(80).strip()

    # parse timestamp
    (day, month, year) = [int(x) for x in re.findall('(\d+)', f.read(8))]
    (hour, minute, sec)= [int(x) for x in re.findall('(\d+)', f.read(8))]
    h['date_time'] = str(datetime.datetime(year + 2000, month, day,
    hour, minute, sec))

    # misc
    subtype = f.read(44)[:5]
    h['EDF+'] = subtype in ['EDF+C', 'EDF+D']
    h['contiguous'] = subtype != 'EDF+D'
    h['n_records'] = int(f.read(8))
    h['record_length'] = float(f.read(8))  # in seconds
    nchannels = h['n_channels'] = int(f.read(4))

    # read channel info
    channels = range(h['n_channels'])
    h['label'] = [f.read(16).strip() for n in channels]
    h['transducer_type'] = [f.read(80).strip() for n in channels]
    h['units'] = [f.read(8).strip() for n in channels]
    h['physical_min'] = np.asarray([float(f.read(8)) for n in channels])
    h['physical_max'] = np.asarray([float(f.read(8)) for n in channels])
    h['digital_min'] = np.asarray([float(f.read(8)) for n in channels])
    h['digital_max'] = np.asarray([float(f.read(8)) for n in channels])
    h['prefiltering'] = [f.read(80).strip() for n in channels]
    h['n_samples_per_record'] = [int(f.read(8)) for n in channels]
    f.read(32 * nchannels)  # reserved

    return h


class BaseEDFReader:
    def __init__(self, file):
        self.file = file


    def read_header(self):
        self.header = h = edf_header(self.file)

        # calculate ranges for rescaling
        self.dig_min = h['digital_min']
        self.phys_min = h['physical_min']
        phys_range = h['physical_max'] - h['physical_min']
        dig_range = h['digital_max'] - h['digital_min']
        assert np.all(phys_range > 0)
        assert np.all(dig_range > 0)
        self.gain = phys_range / dig_range


    def read_raw_record(self):
        '''Read a record with data_2013 and return a list containing arrays with raw
        bytes.
        '''
        result = []
        for nsamp in self.header['n_samples_per_record']:
            samples = self.file.read(nsamp * 2)
            if len(samples) != nsamp * 2:
                raise EDFEndOfData
            result.append(samples)
        return result


    def convert_record(self, raw_record):
        '''Convert a raw record to a (time, signals, events) tuple based on
        information in the header.
        '''
        h = self.header
        dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
        time = float('nan')
        signals = []
        events = []
        for (i, samples) in enumerate(raw_record):
            if h['label'][i] == EVENT_CHANNEL:
                ann = tal(samples)
                time = ann[0][0]
                events.extend(ann[1:])
                # print(i, samples)
                # exit()
            else:
                # 2-byte little-endian integers
                dig = np.fromstring(samples, '<i2').astype(np.float32)
                phys = (dig - dig_min[i]) * gain[i] + phys_min[i]
                signals.append(phys)

        return time, signals, events


    def read_record(self):
        return self.convert_record(self.read_raw_record())


    def records(self):
        '''
        Record generator.
        '''
        try:
            while True:
                yield self.read_record()
        except EDFEndOfData:
            pass


def load_edf(edffile):
    '''Load an EDF+ file.
  Very basic reader for EDF and EDF+ files. While BaseEDFReader does support
  exotic features like non-homogeneous sample rates and loading only parts of
  the stream, load_edf expects a single fixed sample rate for all channels and
  tries to load the whole file.
  Parameters
  ----------
  edffile : file-like object or string
  Returns
  -------
  Named tuple with the fields:
    X : NumPy array with shape p by n.
      Raw recording of n samples in p dimensions.
    sample_rate : float
      The sample rate of the recording. Note that mixed sample-rates are not
      supported.
    sens_lab : list of length p with strings
      The labels of the sensors used to record X.
    time : NumPy array with length n
      The time offset in the recording for each sample.
    annotations : a list with tuples
      EDF+ annotations are stored in (start, duration, description) tuples.
      start : float
        Indicates the start of the event in seconds.
      duration : float
        Indicates the duration of the event in seconds.
      description : list with strings
        Contains (multiple?) descriptions of the annotation event.
  '''
    if isinstance(edffile, basestring):
        with open(edffile, 'rb') as f:
            return load_edf(f)  # convert filename to file

    reader = BaseEDFReader(edffile)
    reader.read_header()

    h = reader.header
    log.debug('EDF header: %s' % h)

      # get sample rate info
    nsamp = np.unique(
        [n for (l, n) in zip(h['label'], h['n_samples_per_record'])
        if l != EVENT_CHANNEL])
    assert nsamp.size == 1, 'Multiple sample rates not supported!'
    sample_rate = float(nsamp[0]) / h['record_length']

    rectime, X, annotations = zip(*reader.records())
    X = np.hstack(X)
    annotations = reduce(operator.add, annotations)
    chan_lab = [lab for lab in reader.header['label'] if lab != EVENT_CHANNEL]

      # create timestamps
    if reader.header['contiguous']:
        time = np.arange(X.shape[1]) / sample_rate
    else:
        reclen = reader.header['record_length']
        within_rec_time = np.linspace(0, reclen, nsamp, endpoint=False)
        time = np.hstack([t + within_rec_time for t in rectime])

    tup = namedtuple('EDF', 'X sample_rate chan_lab time annotations')
    return tup(X, sample_rate, chan_lab, time, annotations)




parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="./shhs1", help="Path to the data")
parser.add_argument("--save_path", type=str, default="./shhs_outputs", help="Path to save preprocessed data")

EPOCH_SEC_SIZE = 30
args = parser.parse_args()
data_dir = os.path.join(args.data_dir, "edfs")
ann_dir = os.path.join(args.data_dir,"annots")
output_dir = os.path.join(args.save_path,"/mid_level_data")
select_ch = 'EEG C4-A1'  #EEG (sec)     C3      A2  #EEG        C4      A1

csv_path = os.path.join(args.data_dir,'/selected_shhs1_files.txt')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

ids = pd.read_csv(csv_path, header=None)
ids = ids[0].values.tolist()

edf_fnames = [os.path.join(data_dir, i + ".edf") for i in ids]
ann_fnames = [os.path.join(ann_dir,  i + "-profusion.xml") for i in ids]

edf_fnames.sort()
ann_fnames.sort()

edf_fnames = np.asarray(edf_fnames)
ann_fnames = np.asarray(ann_fnames)

for file_id in range(len(edf_fnames)):
    if os.path.exists(os.path.join(output_dir, edf_fnames[file_id].split('/')[-1])[:-4]+".npz"):
        continue
    print(edf_fnames[file_id])
    select_ch = 'EEG C4-A1'
    raw = read_raw_edf(edf_fnames[file_id], preload=True, stim_channel=None, verbose=None)
    sampling_rate = raw.info['sfreq']
    ch_type = select_ch.split(" ")[0]    # selecting EEG out of 'EEG C4-A1'
    select_ch = sorted([s for s in raw.info["ch_names"] if ch_type in s]) # this has 2 vals [EEG,EEG(sec)] and selecting 0th index
    print(select_ch)
    raw_ch_df = raw.to_data_frame(scalings=sampling_rate)[select_ch]
    print(raw_ch_df.shape)
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))

    labels = []
    t = ET.parse(ann_fnames[file_id])
    r = t.getroot()
    faulty_File = 0
    for i in range(len(r[4])):
        lbl = int(r[4][i].text)
        if lbl == 4:  # make stages N3, N4 same as N3
            labels.append(3)
        elif lbl == 5:  # Assign label 4 for REM stage
            labels.append(4)
        else:
            labels.append(lbl)
        if lbl > 5:  # some files may contain labels > 5 BUT not the selected ones.
            faulty_File = 1

    if faulty_File == 1:
        print( "============================== Faulty file ==================")
        continue

    labels = np.asarray(labels)

    # Remove movement and unknown stages if any
    raw_ch = raw_ch_df.values
    print(raw_ch.shape)

    # Verify that we can split into 30-s epochs
    if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
        raise Exception("Something wrong")
    n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

    # Get epochs and their corresponding labels
    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)

    print(x.shape)
    print(y.shape)
    assert len(x) == len(y)

    # Select on sleep periods
    w_edge_mins = 30
    nw_idx = np.where(y != 0)[0]
    start_idx = nw_idx[0] - (w_edge_mins * 2)
    end_idx = nw_idx[-1] + (w_edge_mins * 2)
    if start_idx < 0: start_idx = 0
    if end_idx >= len(y): end_idx = len(y) - 1
    select_idx = np.arange(start_idx, end_idx + 1)
    print("Data before selection: {}, {}".format(x.shape, y.shape))
    x = x[select_idx]
    y = y[select_idx]
    print("Data after selection: {}, {}".format(x.shape, y.shape))

    # Saving as numpy files
    filename = os.path.basename(edf_fnames[file_id]).replace(".edf",  ".npz")
    save_dict = {
        "x": x,
        "y": y,
        "fs": sampling_rate
    }
    np.savez(os.path.join(output_dir, filename), **save_dict)
    print(" ---------- Done this file ---------")



seed = 123
np.random.seed(seed)



data_dir = output_dir
output_dir = args.save_path

files = os.listdir(data_dir)
files = np.array([os.path.join(data_dir, i) for i in files])
files.sort()


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

torch.save(data_save, os.path.join(output_dir, "pretext.pt"))

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
torch.save(data_save, os.path.join(output_dir, "train.pt"))

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

torch.save(data_save, os.path.join(output_dir, "val.pt"))
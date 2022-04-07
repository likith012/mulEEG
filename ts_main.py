#%%
import wandb
import numpy as np
import torch
import argparse
from data_preprocessing.dataloader import data_generator
from trainer import sleep_pretrain
from config import Config

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

#%%
parser = argparse.ArgumentParser()
parser.add_argument("--name",type=str,default="mulEEG",help="Name for the saved weights")
parser.add_argument("--data_dir",type=str,default="./SLEEP_data",help="Path to the data")
parser.add_argument("--save_path",type=str,default="./saved_weights",help="Path to save weights")

args = parser.parse_args()

name = args.name
ss_wandb = wandb.init(project='mulEEG Pretrain',name=name,notes='',save_code=True,entity='sleep-staging')
config = Config(ss_wandb)

config.src_path = args.data_dir
config.exp_path = args.save_path

ss_wandb.save('./config.py')
ss_wandb.save('./trainer.py')
ss_wandb.save('./data_preprocessing/*')
ss_wandb.save('./models/*')

print("Loading Data")
dataloader = data_generator(config.src_path,config)
print("Done")

model = sleep_pretrain(config,name,dataloader,ss_wandb)
print('Model Loaded')
ss_wandb.watch([model],log='all',log_freq=500)

model.fit()

ss_wandb.finish()
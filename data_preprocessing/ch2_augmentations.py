#%%
import numpy as np
import torch
from scipy.interpolate import interp1d

def noise_channel(ts, mode, degree):
    """
    Add noise to ts
    
    mode: high, low, both
    degree: degree of noise, compared with range of ts    
    
    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)
        
    """
    len_ts = len(ts)
    num_range = np.ptp(ts)+1e-4 # add a small number for flat signal
    
    ### high frequency noise
    if mode == 'high':
        noise = degree * num_range * (2*np.random.rand(len_ts)-1)
        out_ts = ts + noise
        
    ### low frequency noise
    elif mode == 'low':
        noise = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind='linear')
        noise = f(x_new)
        out_ts = ts + noise
        
    ### both high frequency noise and low frequency noise
    elif mode == 'both':
        noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
        noise2 = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind='linear')
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts

    return out_ts

def jitter(x, config,degree_scale=1):

    #mode = np.random.choice(['high', 'low', 'both', 'no'])
    mode = 'both' 

    ret = []
    for chan in range(x.shape[0]):
        ret.append(noise_channel(x[chan],mode,config.degree*degree_scale))
    ret = np.vstack(ret)
    ret = torch.from_numpy(ret)
    return ret 

def scaling(x,config,degree_scale=2):
    #eprint(x.shape)
    ret = np.zeros_like(x)
    degree = config.degree*(degree_scale+np.random.rand())
    factor = 2*np.random.normal(size=x.shape[1])-1
    factor = 1.5+(2*np.random.rand())+degree*factor
    for i in range(x.shape[0]):
        ret[i]=x[i]*factor
    ret = torch.from_numpy(ret)
    return ret 

def masking(x,config):
    # for each modality we are using different masking
    segments = config.mask_min_points + int(np.random.rand()*(config.mask_max_points-config.mask_min_points))
    points = np.random.randint(0,3000-segments)
    ret = x.detach().clone()
    for i,k in enumerate(x):
        ret[i,points:points+segments] = 0

    return ret

def multi_masking(x, mask_min = 40, mask_max = 10,min_seg= 8, max_seg= 14):
    # Applied idependently to each channel
    fin_masks = []
    segments = min_seg + int(np.random.rand()*(max_seg-min_seg))
    for seg in range(segments):
        fin_masks.append(mask_min + int(np.random.rand() * (mask_max-mask_min)))
    points = np.random.randint(0, 3000-segments,size=segments)
    ret = x.clone()
    for i,k in enumerate(x):
        for seg in range(segments):
            ret[i, points[seg]:points[seg]+fin_masks[seg]] = 0
    return ret

def flip(x,config):
    # horizontal flip
    if np.random.rand() >0.5:
        return torch.tensor(np.flip(x.numpy(),1).copy())
    else:
        return x

def crop(x):
    ret = x.clone()
    if np.random.rand() >0.5:
        l = np.random.randint(1000,2000)
        ret = torch.cat((ret[:,l:],ret[:,:l]),axis=1)
    return ret

def augment(x,config):
    ''' use jitter in every aug to get two different eeg signals '''
    weak_ret = masking(jitter(x,config),config)
    strong_ret = scaling(flip(x,config),config,degree_scale=3)
    return weak_ret,strong_ret

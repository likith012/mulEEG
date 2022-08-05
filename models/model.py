import torch.nn as nn
import torch.nn.functional as F
import torch

from resnet1d import BaseNet
from spectrogram_model import CNNEncoder2D_SLEEP
from config import Config
from typing import Type,Tuple

class encoder(nn.Module):

    """
    Class for the encoder of the model that contains both time encoder and spectrogram encoder 

    Attributes:
    -----------
        config: Config object
            Configuration object specifying the model hyperparameters
    
    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            forward pass of the encoder
    
    """

    def __init__(self,config:Type[Config]):
        super(encoder,self).__init__()
        self.time_model = BaseNet()
        self.attention = attention()

        self.spect_model = CNNEncoder2D_SLEEP(256)
        
    def forward(self, x:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]: 

        time = self.time_model(x)

        time_feats = self.attention(time)

        spect_feats = self.spect_model(x)

        return time_feats,spect_feats


class attention(nn.Module):

    """
    Class for the attention module of the model 

    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            forward pass of the attention module

    """

    def __init__(self):
        super(attention,self).__init__()
        self.att_dim =256
        self.W = nn.Parameter(torch.randn(256, self.att_dim))
        self.V = nn.Parameter(torch.randn(self.att_dim, 1))
        self.scale = self.att_dim**-0.5
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = x.permute(0, 2, 1)
        e = torch.matmul(x, self.W)
        e = torch.matmul(torch.tanh(e), self.V)
        e = e*self.scale
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)
        return x


class projection_head(nn.Module):

    """
    Class for the projection head of the model

    Attributes:
    -----------
        config: Config object
            Configuration object containing hyperparameters

        input_dim: int, optional
            input dimension of the model 
    
    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            forward pass of the projection head

    """

    def __init__(self,config:Type[Config],input_dim:int=256):
        super(projection_head,self).__init__()
        self.config = config
        self.projection_head = nn.Sequential(
                nn.Linear(input_dim,config.tc_hidden_dim),
                nn.BatchNorm1d(config.tc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.tc_hidden_dim,config.tc_hidden_dim))
 
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = x.reshape(x.shape[0],-1)
        x = self.projection_head(x)
        return x

class sleep_model(nn.Module):

    """
    Class for the sleep model

    Attributes:
    -----------
        config: Config object
            Configuration object containing hyperparameters
    
    Methods:
    --------
        forward: torch.Tensor,torch.Tensor -> torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor
            returns the time, spectrogram, and fusion features for both weak and strong inputs
    
    """

    def __init__(self,config:Type[Config]):
        super(sleep_model,self).__init__()
        self.eeg_encoder= encoder(config)
        self.weak_pj1 = projection_head(config)
        self.weak_pj2 = projection_head(config,256*2)
        self.weak_pj3 = projection_head(config)

        self.strong_pj1 = projection_head(config)
        self.strong_pj2 = projection_head(config,256*2)
        self.strong_pj3 = projection_head(config)

        self.wandb = config.wandb

    def forward(self,weak_dat:torch.Tensor,strong_dat:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

        weak_eeg_dat = weak_dat.float()
        strong_eeg_dat = strong_dat.float()
        
        weak_time_feats,weak_spect_feats = self.eeg_encoder(weak_eeg_dat)
        strong_time_feats,strong_spect_feats = self.eeg_encoder(strong_eeg_dat)

        weak_fusion_feats = torch.cat((weak_time_feats,weak_spect_feats),dim=-1)
        weak_fusion_feats = self.weak_pj2(weak_fusion_feats.unsqueeze(1))
        weak_time_feats = self.weak_pj1(weak_time_feats.unsqueeze(1))
        weak_spect_feats = self.weak_pj3(weak_spect_feats.unsqueeze(1))

        strong_fusion_feats = torch.cat((strong_time_feats,strong_spect_feats),dim=-1)
        strong_fusion_feats = self.strong_pj2(strong_fusion_feats.unsqueeze(1))
        strong_time_feats = self.strong_pj1(strong_time_feats.unsqueeze(1))
        strong_spect_feats = self.strong_pj3(strong_spect_feats.unsqueeze(1))
        
        return weak_time_feats,weak_fusion_feats,weak_spect_feats,strong_time_feats,strong_fusion_feats,strong_spect_feats

class contrast_loss(nn.Module):

    """
    Class for the contrast loss 

    Attributes:
    -----------
        config: Config object
            Configuration object containing hyperparameters
    
    Methods:
    --------
        forward: torch.Tensor,torch.Tensor-> torch.Tensor,float,float,float,float

    """

    def __init__(self,config:Type[Config]):

        super(contrast_loss,self).__init__()
        self.model = sleep_model(config)
        self.T = config.temperature
        self.intra_T = config.intra_temperature
        self.bn = nn.BatchNorm1d(config.tc_hidden_dim//2,affine=False)
        self.bn2 = nn.BatchNorm1d(config.tc_hidden_dim//2,affine=False)
        self.bs = config.batch_size
        self.lambd = 0.05
        self.mse = nn.MSELoss(reduction='mean')
        self.wandb = config.wandb

    def loss(self,out_1:torch.Tensor,out_2:torch.Tensor):
        # L2 normalize
        out_1 = F.normalize(out_1, p=2, dim=1)
        out_2 = F.normalize(out_2, p=2, dim=1)

        out = torch.cat([out_1, out_2], dim=0) # 2B, 128
        N = out.shape[0]
        
        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous()) # 2B, 2B
        sim = torch.exp(cov / self.T) # 2B, 2B

        # Negative similarity matrix
        mask = ~torch.eye(N, device = sim.device).bool()
        neg = sim.masked_select(mask).view(N, -1).sum(dim = -1)

        # Positive similarity matrix
        pos = torch.exp(torch.sum(out_1*out_2, dim=-1) / self.T)
        pos = torch.cat([pos, pos], dim = 0) # 2B
        loss = -torch.log(pos / neg).mean()


        return loss
        
    def intra_loss(self,weak_time:torch.Tensor,weak_spect:torch.Tensor,strong_time:torch.Tensor,strong_spect:torch.Tensor)->torch.Tensor:
        weak_time = F.normalize(weak_time,p=2,dim=1)
        strong_time = F.normalize(strong_time,p=2,dim=1)
        strong_spect = F.normalize(strong_spect,p=2,dim=1)
        weak_spect = F.normalize(weak_spect,p=2,dim=1)

        out1 = torch.vstack((weak_time.unsqueeze(0),weak_spect.unsqueeze(0)))
        out2 = torch.vstack((strong_time.unsqueeze(0),strong_spect.unsqueeze(0)))

        out = torch.cat([out1,out2],dim=0) # 4*B*Feat
        N = out.shape[0]

        # similarity matrix
        cov = torch.einsum('abf,dbf->adb',out,out)#/weak_time.shape[-1] # 4*4*B
        sim = torch.exp(cov/self.intra_T)


        # negtive similarity matrix
        mask = ~torch.eye(N,device=sim.device).bool()
        neg = sim[mask].view(N,N-1,weak_time.shape[0]).sum(dim=1)

        # positive similarity matrix
        pos = torch.exp(torch.sum(out1*out2,dim=-1)/self.intra_T)
        pos = torch.cat([pos,pos],dim=0)
        loss = -torch.log(pos/neg)
        loss = loss.mean()
        return loss

    def forward(self,weak:torch.Tensor,strong:torch.Tensor)->Tuple[torch.Tensor,float,float,float,float]:

        weak_time_feats,weak_fusion_feats,weak_spect_feats,strong_time_feats,strong_fusion_feats,strong_spect_feats= self.model(weak,strong)

        l1 = self.loss(weak_time_feats,strong_time_feats)
        l2 = self.loss(weak_fusion_feats,strong_fusion_feats)
        l3 = self.loss(weak_spect_feats,strong_spect_feats)
        intra_loss = self.intra_loss(weak_time_feats,weak_spect_feats,strong_time_feats,strong_spect_feats)

        tot_loss = l1+l2+l3+intra_loss

        return tot_loss,l1.item(),l2.item(),l3.item(),intra_loss.item()


class ft_loss(nn.Module):

    """
    Class for the linear evaluation module for time encoder features

    Attributes:
    ----------
        chkpoint_pth: str
            Path to the checkpoint file
        config: Config object
            Configuration object containing hyperparameters
        device: str
            Device to use for training
    
    Methods:
    --------
        forward: torch.Tensor-> torch.Tensor
            Computes the forward pass
    
    """

    def __init__(self,chkpoint_pth:str,config:Type[Config],device:str):

        super(ft_loss,self).__init__()

        self.eeg_encoder = encoder(config)
        chkpoint = torch.load(chkpoint_pth,map_location=device)
        eeg_dict = chkpoint['eeg_model_state_dict']
        self.eeg_encoder.load_state_dict(eeg_dict)

        for p in self.eeg_encoder.parameters():
            p.requires_grad=False

        self.lin = nn.Linear(256,5)

    def forward(self,time_dat:torch.Tensor)->torch.Tensor:

        time_feats,_ = self.eeg_encoder(time_dat)
        x = self.lin(time_feats)

        return x 

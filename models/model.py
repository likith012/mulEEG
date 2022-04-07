#%%
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
#%%
from .resnet1d import BaseNet

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2) 
        self.downsample = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out

class CNNEncoder2D_SLEEP(nn.Module):
    def __init__(self, n_dim):
        super(CNNEncoder2D_SLEEP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(6, 8, 2, True, False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 32, 2, True, True)
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.Linear(128, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

        self.sup = nn.Sequential(
            nn.Linear(128, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 5, bias=True),
        )

        self.byol_mapping = nn.Sequential(
            nn.Linear(128, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    def torch_stft(self, X_train):
        signal = []

        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 * 1 // 4,
                center = False,
                onesided = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3) # amplitude (B, 2, H, W)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3) # phase (B, 2, H, W)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1) # (B, 4, H, W)
        # 0, 1 for 1st channel, 2, 3 for 2nd channel

    def forward(self, x, simsiam=False, mid=False, byol=False, sup=False):
        # Inputs -> (B, 2, 3000)
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # (B, 32, 4, 1)

        x = x.reshape(x.shape[0], -1)

        if sup:
            return self.sup(x)
        elif simsiam:
            return x, self.fc(x)
        elif mid:
            return x
        elif byol:
            x = self.fc(x)
            x = self.byol_mapping(x)
            return x
        else:
            x = self.fc(x)
            return x
#%%
class encoder(nn.Module):

    def __init__(self,config):
        super(encoder,self).__init__()
        self.time_model = BaseNet()
        self.attention = attention(config)

        self.spect_model = CNNEncoder2D_SLEEP(256)
        
    def forward(self, x): 

        time = self.time_model(x)

        time_feats = self.attention(time)

        spect_feats = self.spect_model(x)

        return time_feats,spect_feats


class attention(nn.Module):
    def __init__(self,config):
        super(attention,self).__init__()
        self.att_dim =256
        self.W = nn.Parameter(torch.randn(256, self.att_dim))
        self.V = nn.Parameter(torch.randn(self.att_dim, 1))
        self.scale = self.att_dim**-0.5
    def forward(self,x):
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

    def __init__(self,config,input_dim=256):
        super(projection_head,self).__init__()
        self.config = config
        self.projection_head = nn.Sequential(
                nn.Linear(input_dim,config.tc_hidden_dim),
                nn.BatchNorm1d(config.tc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.tc_hidden_dim,config.tc_hidden_dim))
 
    def forward(self,x):
        x = x.reshape(x.shape[0],-1)
        x = self.projection_head(x)
        return x

class sleep_model(nn.Module):

    def __init__(self,config):
        super(sleep_model,self).__init__()
        self.eeg_encoder= encoder(config)
        self.weak_pj1 = projection_head(config)
        self.weak_pj2 = projection_head(config,256*2)
        self.weak_pj3 = projection_head(config)

        self.strong_pj1 = projection_head(config)
        self.strong_pj2 = projection_head(config,256*2)
        self.strong_pj3 = projection_head(config)

        self.wandb = config.wandb

    def forward(self,weak_dat,strong_dat):
        weak_eeg_dat = weak_dat.float()

        strong_eeg_dat = strong_dat.float()
        
        weak_time_feats,weak_spect_feats = self.eeg_encoder(weak_eeg_dat)
        strong_time_feats,strong_spect_feats = self.eeg_encoder(strong_eeg_dat)

        #if self.wandb!=None:
        #    self.wandb.log({'Weak Time std: ': np.std(weak_time_feats.detach().cpu().numpy(),axis=-1).mean(),
        #        'Weak Spect std: ': np.std(weak_spect_feats.detach().cpu().numpy(),axis=-1).mean(),
        #        'Strong Time std: ': np.std(strong_time_feats.detach().cpu().numpy(),axis=-1).mean(),
        #        'Strong Spect std: ': np.std(strong_spect_feats.detach().cpu().numpy(),axis=-1).mean()})

        weak_fusion_feats = torch.cat((weak_time_feats,weak_spect_feats),dim=-1)
        weak_fusion_feats = self.weak_pj2(weak_fusion_feats.unsqueeze(1))
        weak_time_feats = self.weak_pj1(weak_time_feats.unsqueeze(1))
        weak_spect_feats = self.weak_pj3(weak_spect_feats.unsqueeze(1))

        strong_fusion_feats = torch.cat((strong_time_feats,strong_spect_feats),dim=-1)
        strong_fusion_feats = self.strong_pj2(strong_fusion_feats.unsqueeze(1))
        strong_time_feats = self.strong_pj1(strong_time_feats.unsqueeze(1))
        strong_spect_feats = self.strong_pj3(strong_spect_feats.unsqueeze(1))
        

        #if self.wandb!=None:
            #self.wandb.log({'weak pj1 std: ': np.std(weak_time_feats.detach().cpu().numpy(),axis=-1).mean(),
            #    'weak pj3 std: ': np.std(weak_spect_feats.detach().cpu().numpy(),axis=-1).mean(),
            #    'weak pj2 std: ': np.std(weak_fusion_feats.detach().cpu().numpy(),axis=-1).mean(),
            #    'strong pj1 std: ': np.std(strong_time_feats.detach().cpu().numpy(),axis=-1).mean(),
            #    'strong pj3 std: ': np.std(strong_spect_feats.detach().cpu().numpy(),axis=-1).mean(),
            #    'strong pj2 std: ': np.std(strong_fusion_feats.detach().cpu().numpy(),axis=-1).mean()})

        return weak_time_feats,weak_fusion_feats,weak_spect_feats,strong_time_feats,strong_fusion_feats,strong_spect_feats

class contrast_loss(nn.Module):

    def __init__(self,config):

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

    def loss(self,out_1,out_2):
        # L2 normalize
        out_1 = F.normalize(out_1, p=2, dim=1)
        out_2 = F.normalize(out_2, p=2, dim=1)

        out = torch.cat([out_1, out_2], dim=0) # 2B, 128
        N = out.shape[0]
        
        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous()) # 2B, 2B
        sim = torch.exp(cov / self.T) # 2B, 2B
        #sim = torch.exp(cov) # 2B, 2B

        # Negative similarity matrix
        mask = ~torch.eye(N, device = sim.device).bool()
        neg = sim.masked_select(mask).view(N, -1).sum(dim = -1)

        # Positive similarity matrix
        pos = torch.exp(torch.sum(out_1*out_2, dim=-1) / self.T)
        #pos = torch.exp(torch.sum(out_1*out_2, dim=-1))
        pos = torch.cat([pos, pos], dim = 0) # 2B
        loss = -torch.log(pos / neg).mean()


        return loss
#%%
    def intra_loss(self,weak_time,weak_spect,strong_time,strong_spect):
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
        #sim = torch.exp(cov)


        # negtive similarity matrix
        mask = ~torch.eye(N,device=sim.device).bool()
        neg = sim[mask].view(N,N-1,weak_time.shape[0]).sum(dim=1)

        # positive similarity matrix
        pos = torch.exp(torch.sum(out1*out2,dim=-1)/self.intra_T)
        #pos = torch.exp(torch.sum(out1*out2,dim=-1))
        pos = torch.cat([pos,pos],dim=0)
        loss = -torch.log(pos/neg)
        loss = loss.mean()
        return loss
#%%

    def forward(self,weak,strong,epoch):
        weak_time_feats,weak_fusion_feats,weak_spect_feats,strong_time_feats,strong_fusion_feats,strong_spect_feats= self.model(weak,strong)
        l1 = self.loss(weak_time_feats,strong_time_feats)
        l2 = self.loss(weak_fusion_feats,strong_fusion_feats)
        l3 = self.loss(weak_spect_feats,strong_spect_feats)
        intra_loss = self.intra_loss(weak_time_feats,weak_spect_feats,strong_time_feats,strong_spect_feats)
        tot_loss = l1+l2+l3+intra_loss

        return tot_loss,l1.item(),l2.item(),l3.item(),intra_loss.item()


#%%
class ft_loss(nn.Module):

    def __init__(self,chkpoint_pth,config,device):

        super(ft_loss,self).__init__()
        self.eeg_encoder = encoder(config)
        
        chkpoint = torch.load(chkpoint_pth,map_location=device)
        eeg_dict = chkpoint['eeg_model_state_dict']

        self.eeg_encoder.load_state_dict(eeg_dict)

        for p in self.eeg_encoder.parameters():
            p.requires_grad=False
        self.lin = nn.Linear(256*2,5)

    def forward(self,time_dat):

        time_feats,spect_feats = self.eeg_encoder(time_dat)
        feats = torch.cat((time_feats,spect_feats),dim=-1)
        x = self.lin(feats)
        return x 


""" Spectrogram model based on the ResNet1D with inputs as spectrograms (phase + amplitude).

More details on the paper at Yang, Chaoqi, et al. "Self-supervised EEG Representation Learning for Automatic Sleep Staging." 
Refer to the link at arXiv preprint https://arxiv.org/abs/2110.15278.

This file can also be imported as a module and contains the following functions:

    * ResBlock - A Simple ResNet Block with 2 convolutional layers and a skip connection.
    * CNNEncoder2D_SLEEP - An encoder for the spectrogram model.

"""

import torch.nn as nn
import torch


class ResBlock(nn.Module):
    """
    A Simple ResNet Block with 2 convolutional layers and a skip connection

    Attributes:
    -----------
    in_channels: int
        number of input channels
    out_channels: int
        number of output channels
    stride: int, optional
        stride of the convolutional layers
    downsample: bool, optional
        whether to downsample the input
    pooling: bool, optional
        whether to pool the input
    
    Methods:
    --------
    forward(x)
        forward pass of the block
        
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
        pooling: bool = False,
    ) -> None:
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the block."""

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
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
    """ An encoder for the spectrogram model.

    Attributes:
    -----------
    n_dim: int
        Number of dimensions of the latent space
    
    Methods:
    --------
    torch_dtft(X_train)
        Compute the STFT of the input
    forward(x)
        Forward pass of the block

    References:
    -----------
        Yang, Chaoqi, et al. "Self-supervised EEG Representation Learning for Automatic Sleep Staging." arXiv preprint arXiv:2110.15278 (2021).

    """

    def __init__(self, n_dim: int) -> None:
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

    def torch_stft(self, X_train: torch.Tensor) -> torch.Tensor:
        """ Compute the STFT of the input."""

        signal = []

        for s in range(X_train.shape[1]):
            spectral = torch.stft(
                X_train[:, s, :],
                n_fft=256,
                hop_length=256 * 1 // 4,
                center=False,
                onesided=True,
                return_complex=False,
            )
            signal.append(spectral)

        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat(
            [
                torch.log(torch.abs(signal1) + 1e-8),
                torch.log(torch.abs(signal2) + 1e-8),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the block."""

        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x

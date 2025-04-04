import torch
import torch.nn as nn

from eqnr.nn.modules import ConvBlock2d, ConvBlock3d, ConvTransposeBlock2d, ConvTransposeBlock3d, ResBlock2d, ResBlock3d

__all__ = [
    "Decode2d",
    "Decode3d",
]


class _DecodeNd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pre_activation: bool=True,
        dim: int=None,
        device=None
    ) -> None:
        super().__init__()
        assert dim in [2,3], "dim must be 2 (Conv2d) or 3 (Conv3d)"

        self.device = device
        self.in_channels= in_channels
        self.out_channels = out_channels  #128
        self.dim = dim

        _Conv = {2: nn.Conv2d, 3: nn.Conv3d}[self.dim]

        #_ConvTranspose = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[self.dim]

        _ResBlock = {2: ResBlock2d, 3: ResBlock3d}[self.dim]
        _ConvBlock = {2: ConvBlock2d, 3: ConvBlock3d}[self.dim]
        _ConvTransposeBlock = {2:  ConvTransposeBlock2d, 3:  ConvTransposeBlock3d}[self.dim]

        if dim == 3:
            self.layers = nn.Sequential(
                _ResBlock(self.in_channels, 64, pre_activation=pre_activation),
                _ResBlock(64, 64, pre_activation=pre_activation),
                
                _ConvBlock(64, 128, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
                _ResBlock(128, 128, pre_activation=pre_activation),
                _ResBlock(128, 128, pre_activation=pre_activation),
                
                _ConvTransposeBlock(128, 32, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
                _ResBlock(32, 32, pre_activation=pre_activation),
                _ResBlock(32, self.out_channels, pre_activation=pre_activation),
            ).to(self.device)
        else: # dim == 2
            self.layers = nn.Sequential(
                _ResBlock(self.in_channels, 128, pre_activation=pre_activation),
                _ResBlock(128, 128, pre_activation=pre_activation),
                
                _ConvBlock(128, 256, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
                _ResBlock(256, 256, pre_activation=pre_activation),
                
                _ConvTransposeBlock(256, 128, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
                _ResBlock(128, 128, pre_activation=pre_activation),
                
                _ConvTransposeBlock(128, 128, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
                _ResBlock(128, 128, pre_activation=pre_activation),
                
                _ConvTransposeBlock(128, 64, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation), 
                _ResBlock(64, 64, pre_activation=pre_activation),
                _ResBlock(64, 64, pre_activation=pre_activation),

                _ConvBlock(64, self.out_channels, kernel_size=1, stride=1, padding=0) if pre_activation else _Conv(64, self.out_channels, kernel_size=1, stride=1, padding=0)
            ).to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Decode2d(_DecodeNd):
    def __init__(self, in_channels, out_channels, pre_activation = True, device=None):
        super().__init__(in_channels, out_channels, pre_activation, dim=2, device=device)


class Decode3d(_DecodeNd):
    def __init__(self, in_channels, out_channels, pre_activation = True, device=None):
        super().__init__(in_channels, out_channels, pre_activation, dim=3, device=device)
import torch
import torch.nn as nn

from eqnr.nn.modules import ConvBlock2d, ConvBlock3d, ConvTransposeBlock2d, ConvTransposeBlock3d, ResBlock2d, ResBlock3d

__all__ = [
    "Encode2d",
    "Encode3d",
]


class _EncodeNd(nn.Module):
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

        if dim == 2:
            self.layers = nn.Sequential(
                ConvBlock2d(self.in_channels, 64, kernel_size=1, stride=1, padding=0, pre_activation=False),

                ResBlock2d(64, 64, pre_activation=pre_activation),
                ResBlock2d(64, 64, pre_activation=pre_activation),
                ConvBlock2d(64, 128, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),

                ResBlock2d(128, 128, pre_activation=pre_activation),
                ConvBlock2d(128, 128, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),

                ResBlock2d(128, 128, pre_activation=pre_activation),
                ConvBlock2d(128, 256, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
                
                ResBlock2d(256, 256, pre_activation=pre_activation),
                ConvTransposeBlock2d(256, 128, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),

                ResBlock2d(128, 128, pre_activation=pre_activation),
                ResBlock2d(128, self.out_channels, pre_activation=pre_activation)
            ).to(self.device)
        else: # dim == 3
             self.layers = nn.Sequential(
                ConvBlock3d(self.in_channels, 32, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
                
                ResBlock3d(32, 32, pre_activation=pre_activation),
                ResBlock3d(32, 32, pre_activation=pre_activation),
                ConvBlock3d(32, 128, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),
                
                ResBlock3d(128, 128, pre_activation=pre_activation),
                ResBlock3d(128, 128, pre_activation=pre_activation),
                ConvTransposeBlock3d(128, 64, kernel_size=4, stride=2, padding=1, pre_activation=pre_activation),

                ResBlock3d(64, 64, pre_activation=pre_activation),
                ResBlock3d(64, self.out_channels, pre_activation=pre_activation)
             )
            
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Encode2d(_EncodeNd):
    def __init__(self, in_channels, out_channels, pre_activation = True, device=None):
        super().__init__(in_channels, out_channels, pre_activation, dim=2, device=device)


class Encode3d(_EncodeNd):
    def __init__(self, in_channels, out_channels, pre_activation = True, device=None):
        super().__init__(in_channels, out_channels, pre_activation, dim=3, device=device)
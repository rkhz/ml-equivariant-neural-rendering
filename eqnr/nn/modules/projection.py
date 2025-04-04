import torch.nn as nn
import einops.layers.torch as einops_nn

from .conv_block import ConvBlock2d, ConvBlock3d

__all__ = [
    "Project2DTo3D",
    "Project3DTo2D"
]


class Project2DTo3D(nn.Module):
    """
    Notes:
        This layer is inspired by the Projection Unit from
        https://arxiv.org/abs/1806.06575.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        pre_activation: bool=True, 
        device=None
    ) -> None:
        super(Project2DTo3D, self).__init__()
        super().__init__()
        
        self.device = device
        self.in_channels= in_channels 
        self.out_channels = out_channels  #128
        
        self.layers = nn.Sequential(
            ConvBlock2d(self.in_channels, 256, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
            ConvBlock2d(256, 512, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
            ConvBlock2d(512,1024, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
            einops_nn.Rearrange('b (c d) h w -> b c d h w', c=self.out_channels),
            ConvBlock3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, pre_activation=pre_activation),
        )

    def forward(self, x):
        return self.layers(x)



class Project3DTo2D(nn.Module):
    """
    Notes:
        This layer is inspired by the Projection Unit from
        https://arxiv.org/abs/1806.06575.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        pre_activation: bool=True, 
        device=None
    ) -> None:
        super(Project3DTo2D, self).__init__()
        super().__init__()
        
        self.device = device
        self.in_channels= in_channels 
        self.out_channels = out_channels  #128
        
        self.layers = nn.Sequential(
            ConvBlock3d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, pre_activation=pre_activation),
            einops_nn.Rearrange('b c d h w -> b (c d) h w', c=self.in_channels),
            ConvBlock2d(1024, 512, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),   # same as doing Conv actually 
            ConvBlock2d( 512, 256, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),   # same as doing Conv actually 
            ConvBlock2d( 256, 128, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation)    # same as doing Conv actually 
        )

    def forward(self, x):
        return self.layers(x)

    